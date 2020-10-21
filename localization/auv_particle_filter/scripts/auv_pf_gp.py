#!/usr/bin/env python3

# Standard dependencies
import math
import rospy
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as rot

from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from geometry_msgs.msg import Transform, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Header

from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf.transformations import translation_matrix, translation_from_matrix
from tf.transformations import quaternion_matrix, quaternion_from_matrix
from tf.transformations import rotation_matrix, rotation_from_matrix

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

# For sim mbes action client
import actionlib
from auv_2_ros.msg import MbesSimGoal, MbesSimAction, MbesSimResult
from auv_particle_gp import Particle, matrix_from_tf, pcloud2ranges, pack_cloud
from resampling import residual_resample, naive_resample, systematic_resample, stratified_resample

# Auvlib
from auvlib.bathy_maps import base_draper
from auvlib.data_tools import csv_data

from scipy.ndimage.filters import gaussian_filter

# gpytorch
from bathy_gps import process, gp

class auv_pf(object):

    def __init__(self):
        # Read necessary parameters
        self.pc = rospy.get_param('~particle_count', 10) # Particle Count
        self.map_frame = rospy.get_param('~map_frame', 'map') # map frame_id
        odom_frame = rospy.get_param('~odom_frame', 'odom')
        meas_model_as = rospy.get_param('~mbes_as', '/mbes_sim_server') # map frame_id
        self.beams_num = rospy.get_param("~num_beams_sim", 20)
        self.beams_real = rospy.get_param("~n_beams_mbes", 512)
        self.mbes_angle = rospy.get_param("~mbes_open_angle", np.pi/180. * 60.)

        # Initialize tf listener
        tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tfBuffer)
        try:
            rospy.loginfo("Waiting for transforms")
            mbes_tf = tfBuffer.lookup_transform('hugin/base_link', 'hugin/mbes_link',
                                                rospy.Time(0), rospy.Duration(20))
            self.base2mbes_mat = matrix_from_tf(mbes_tf)

            m2o_tf = tfBuffer.lookup_transform(self.map_frame, odom_frame,
                                               rospy.Time(0), rospy.Duration(20))
            self.m2o_mat = matrix_from_tf(m2o_tf)

            rospy.loginfo("Transforms locked - pf node")
        except:
            rospy.loginfo("ERROR: Could not lookup transform from base_link to mbes_link")

        # Read covariance values
        meas_cov = float(rospy.get_param('~measurement_covariance', 0.01))
        cov_string = rospy.get_param('~motion_covariance')
        cov_string = cov_string.replace('[','')
        cov_string = cov_string.replace(']','')
        cov_list = list(cov_string.split(", "))
        motion_cov = list(map(float, cov_list))

        cov_string = rospy.get_param('~init_covariance')
        cov_string = cov_string.replace('[','')
        cov_string = cov_string.replace(']','')
        cov_list = list(cov_string.split(", "))
        init_cov = list(map(float, cov_list))

        cov_string = rospy.get_param('~resampling_noise_covariance')
        cov_string = cov_string.replace('[','')
        cov_string = cov_string.replace(']','')
        cov_list = list(cov_string.split(", "))
        self.res_noise_cov = list(map(float, cov_list))

        # Initialize list of particles
        self.particles = np.empty(self.pc, dtype=object)

        for i in range(self.pc):
            self.particles[i] = Particle(self.beams_num, self.pc, i, self.base2mbes_mat,
                                         self.m2o_mat, init_cov=init_cov, meas_cov=meas_cov,
                                         process_cov=motion_cov)

        self.time = None
        self.old_time = None
        self.pred_odom = None
        self.latest_mbes = PointCloud2()
        self.prev_mbes = PointCloud2()
        self.poses = PoseArray()
        self.poses.header.frame_id = odom_frame
        self.avg_pose = PoseWithCovarianceStamped()
        self.avg_pose.header.frame_id = odom_frame

        # Initialize particle poses publisher
        pose_array_top = rospy.get_param("~particle_poses_topic", '/particle_poses')
        self.pf_pub = rospy.Publisher(pose_array_top, PoseArray, queue_size=10)

        # Initialize average of poses publisher
        avg_pose_top = rospy.get_param("~average_pose_topic", '/average_pose')
        self.avg_pub = rospy.Publisher(avg_pose_top, PoseWithCovarianceStamped, queue_size=10)

        # Establish subscription to odometry message (intentionally last)
        odom_top = rospy.get_param("~odometry_topic", 'odom')
        rospy.Subscriber(odom_top, Odometry, self.odom_callback)

        # Expected meas of PF outcome at every time step
        pf_mbes_top = rospy.get_param("~average_mbes_topic", '/avg_mbes')
        self.pf_mbes_pub = rospy.Publisher(pf_mbes_top, PointCloud2, queue_size=1)

        self.stats = rospy.Publisher('/pf/n_eff', Float32, queue_size=10)
        rospy.loginfo("Particle filter class successfully created")

        mbes_pc_top = rospy.get_param("~particle_sim_mbes_topic", '/sim_mbes')
        self.pcloud_pub = rospy.Publisher(mbes_pc_top, PointCloud2, queue_size=10)
        
        # Load mesh
        print("Loading data")
        svp_path = rospy.get_param('~sound_velocity_prof')
        mesh_path = rospy.get_param('~mesh_path')
        sound_speeds = csv_data.csv_asvp_sound_speed.parse_file(svp_path)
        data = np.load(mesh_path)
        V, F, bounds = data['V'], data['F'], data['bounds'] 
        print("Mesh loaded")

        # Create draper
        self.draper = base_draper.BaseDraper(V, F, bounds, sound_speeds)
        self.draper.set_ray_tracing_enabled(False)            
        print("draper created")
 
        # Load GP
        gp_path = rospy.get_param("~gp_path", 'gp.path')
        self.gp = gp.SVGP.load(500, gp_path)

        # Action server for MBES pings sim (necessary to be able to use UFO maps as well)
        self.as_ping = actionlib.SimpleActionServer('/mbes_sim_server', MbesSimAction, 
                                                    execute_cb=self.mbes_as_cb)

        # Subscription to real mbes pings 
        mbes_pings_top = rospy.get_param("~mbes_pings_topic", 'mbes_pings')
        rospy.Subscriber(mbes_pings_top, PointCloud2, self.mbes_real_cb)
        
        # Create expected MBES beams directions
        angle_step = self.mbes_angle/self.beams_num
        self.beams_dir = []
        for i in range(0, self.beams_num):
            roll_step = rotation_matrix(-self.mbes_angle/2.
                                        + angle_step * i, (1,0,0))[0:3, 0:3]
            rot = roll_step[:,2]
            self.beams_dir.append(rot/np.linalg.norm(rot))

        self.update_rviz()
        rospy.spin()

    def gp_sampling(self, p, R):
        h = 40. # Depth of the field of view
        b = h * np.cos(self.mbes_angle/2.)
        n = 60  # Number of sampling points

        # Triangle vertices 
        Rxmin = rotation_matrix(-self.mbes_angle/2., (1,0,0))[0:3, 0:3]
        Rxmax = rotation_matrix(self.mbes_angle/2., (1,0,0))[0:3, 0:3]
        Rxmin = np.matmul(R, Rxmin)
        Rxmax = np.matmul(R, Rxmax)
        
        p2 = p + Rxmax[:,2]/np.linalg.norm(Rxmax[:,2]) * b
        p3 = p + Rxmin[:,2]/np.linalg.norm(Rxmin[:,2]) * b
        
        # Sampling step
        i_range = np.linalg.norm(p3 - p2)/n

        # Across ping direction
        direc = ((p3-p2)/np.linalg.norm(p3-p2)) 
        sampling_points = []
        for i in range(0, n):
            sampling_points.append(p2 + direc * i_range * i)

        # Sample GP here
        mu, sigma = self.gp.sample(np.asarray(sampling_points)[:, 0:2])
        mu_array = np.array([mu])
        
        # Concatenate sampling points x,y with sampled z
        mbes_gp = np.concatenate((np.asarray(sampling_points)[:, 0:2], 
                                  mu_array.T), axis=1)
        #  print(mbes_gp)
        return mbes_gp


    def gp_ray_tracing(self, r_mbes, p, gp_samples, beams_num):

        # Transform points to MBES frame to perform ray tracing on 2D
        rot_inv = r_mbes.transpose()
        p_inv = rot_inv.dot(p)
        gp_samples = [rot_inv.dot(beam) - p_inv for beam in gp_samples]
        
        # TODO: this can be faster
        exp_meas = []
        for m in range(0, len(self.beams_dir)):
            # No need to iterate over all the points for each beam
            for i in range(1, len(gp_samples)):
                v1 = -gp_samples[i-1]
                v2 = gp_samples[i] - gp_samples[i-1]
                v3 = [self.beams_dir[m][0], -self.beams_dir[m][2], self.beams_dir[m][1]]

                t1 = np.linalg.norm(np.cross(v2, v1))/v2.dot(v3)
                if (t1 > 0):
                    t2 = v1.dot(v3)/v2.dot(v3)
                    if(t2 > 0 and t2 < 1):
                        exp_meas.append(self.beams_dir[m] * t1)
                        break
        
        # Transform back to map frame
        mbes_gp = np.asarray(exp_meas)
        mbes_gp = [r_mbes.dot(beam) + p for beam in mbes_gp]

        return mbes_gp

    def mbes_real_cb(self, msg):
        self.latest_mbes = msg

    # Action server to simulate MBES for the sim AUV
    def mbes_as_cb(self, goal):

        #  print("MBES sim goal received")
        # Unpack goal
        p_auv = [goal.mbes_pose.transform.translation.x, 
                 goal.mbes_pose.transform.translation.y, 
                 goal.mbes_pose.transform.translation.z]
        r_mbes = quaternion_matrix([goal.mbes_pose.transform.rotation.x,
                                    goal.mbes_pose.transform.rotation.y,
                                    goal.mbes_pose.transform.rotation.z,
                                    goal.mbes_pose.transform.rotation.w])[0:3, 0:3]

        # Rotate roll 180 degrees for base frame
        R = rot.from_euler("zyx", [0., 0., np.pi]).as_dcm() 
        r_base = r_mbes.dot(R)
                
        # IGL sim ping
        mbes = self.draper.project_mbes(np.asarray(p_auv), r_base,
                                        goal.beams_num.data, self.mbes_angle)

        #  mbes[:, -2] = gaussian_filter(mbes[:,-2], sigma=0.5)

        # Pack result
        mbes_cloud = pack_cloud(self.map_frame, mbes)
        result = MbesSimResult()
        result.sim_mbes = mbes_cloud
        self.as_ping.set_succeeded(result)

        self.latest_mbes = result.sim_mbes 

    def odom_callback(self, odom_msg):
        self.time = odom_msg.header.stamp.to_sec()
        if self.old_time and self.time > self.old_time:
            # Motion prediction
            self.predict(odom_msg)    
            
            if self.latest_mbes.header.stamp > self.prev_mbes.header.stamp:    
                # Measurement update if new one received
                weights = self.update(self.latest_mbes, odom_msg)
                self.prev_mbes = self.latest_mbes
                
                # Particle resampling
                self.resample(weights)

            self.update_rviz()
        self.old_time = self.time

    def predict(self, odom_t):
        dt = self.time - self.old_time
        for i in range(0, self.pc):
            self.particles[i].motion_pred(odom_t, dt)

    def update(self, real_mbes, odom):
        # Compute AUV MBES ping ranges
        particle_tf = Transform()
        particle_tf.translation = odom.pose.pose.position
        particle_tf.rotation = odom.pose.pose.orientation
        tf_mat = matrix_from_tf(particle_tf)
        m2auv = np.matmul(self.m2o_mat, np.matmul(tf_mat, self.base2mbes_mat))
        real_mbes_ranges = pcloud2ranges(real_mbes, m2auv)

        # Measurement update of each particle
        R = rot.from_euler("zyx", [0., 0., np.pi]).as_dcm() 
        for i in range(0, self.pc):
            # Rotate roll 180 degrees for base frame
            p_part, r_mbes = self.particles[i].get_p_pose();
            r_base = r_mbes.dot(R)
            
            # Sample GP points
            #  gp_samples = self.gp_sampling(p_part, r_base)

            #  # Perform raytracing over segments between GP sampled points
            #  exp_mbes = self.gp_ray_tracing(r_mbes, p_part, gp_samples, self.beams_num)
                   
            # MBES sim on IGL
            #  Rotate pitch 90 degrees for MBES z axis to point downwards
            R = rot.from_euler("zyx", [0., 0., np.pi]).as_dcm() 
            exp_mbes = self.draper.project_mbes(np.asarray(p_part), r_mbes.dot(R),
                                                self.beams_num, self.mbes_angle)
            
            # Publish (for visualization)
            mbes_pcloud = pack_cloud(self.map_frame, exp_mbes)
            self.pcloud_pub.publish(mbes_pcloud)

            self.particles[i].compute_weight(exp_mbes, real_mbes_ranges, True)
        
        weights = []
        for i in range(self.pc):
            weights.append(self.particles[i].w)

        self.miss_meas = weights.count(1.e-50)
        weights_array = np.asarray(weights)
        # Add small non-zero value to avoid hitting zero
        weights_array += 1.e-30

        return weights_array

    def ping2ranges(self, point_cloud):
        ranges = []
        cnt = 0
        for p in pc2.read_points(point_cloud, 
                                 field_names = ("x", "y", "z"), skip_nans=True):
            ranges.append(np.linalg.norm(p[-2:]))
            #  if cnt == 0:
                #  print "Beam 0 original ", p
            #  cnt += 1
        return np.asarray(ranges)

    def resample(self, weights):

        #  print "-------------"
        # Normalize weights
        weights /= weights.sum()
        #  print (weights)

        N_eff = self.pc
        if weights.sum() == 0.:
            rospy.loginfo("All weights zero!")
        else:
            N_eff = 1/np.sum(np.square(weights))

        print ("N_eff ", N_eff)
        #  print "Missed meas ", self.miss_meas
        if self.miss_meas < self.pc/2.:
            self.stats.publish(np.float32(N_eff)) 
        
        # Resampling?
        if N_eff < self.pc/2. and self.miss_meas < self.pc/2.:
            indices = residual_resample(weights)
            #  print "Resampling "
            #  print indices
            keep = list(set(indices))
            lost = [i for i in range(self.pc) if i not in keep]
            dupes = indices[:].tolist()
            for i in keep:
                dupes.remove(i)

            self.reassign_poses(lost, dupes)
            # Add noise to particles
            for i in range(self.pc):
                self.particles[i].add_noise(self.res_noise_cov)

        #  else:
            #  rospy.loginfo('Number of effective particles high - not resampling')

    def reassign_poses(self, lost, dupes):
        for i in range(len(lost)):
            # Faster to do separately than using deepcopy()
            self.particles[lost[i]].p_pose[0] = self.particles[dupes[i]].p_pose[0]
            self.particles[lost[i]].p_pose[1] = self.particles[dupes[i]].p_pose[1]
            self.particles[lost[i]].p_pose[2] = self.particles[dupes[i]].p_pose[2]
            self.particles[lost[i]].p_pose[3] = self.particles[dupes[i]].p_pose[3]
            self.particles[lost[i]].p_pose[4] = self.particles[dupes[i]].p_pose[4]
            self.particles[lost[i]].p_pose[5] = self.particles[dupes[i]].p_pose[5]
    
    def average_pose(self, pose_list):

        poses_array = np.array(pose_list)
        ave_pose = poses_array.mean(axis = 0)
        self.avg_pose.pose.pose.position.x = ave_pose[0]
        self.avg_pose.pose.pose.position.y = ave_pose[1]
        self.avg_pose.pose.pose.position.z = ave_pose[2]
        roll  = ave_pose[3]
        pitch = ave_pose[4]
        """
        Average of yaw angles creates
        issues when heading towards pi because pi and
        negative pi are next to eachother, but average
        out to zero (opposite direction of heading)
        """
        yaws = poses_array[:,5]
        #  print yaws
        if np.abs(yaws).min() > math.pi/2:
            yaws[yaws < 0] += 2*math.pi
        yaw = yaws.mean()

        #  yaw = ((yaw) + 2 * np.pi) % (2 * np.pi)
        #  for yaw_i in yaws:
            #  yaw_i = (yaw_i + np.pi) % (2 * np.pi) - np.pi
        #  yaw = (yaws.mean() + np.pi) % (2 * np.pi) - np.pi

        self.avg_pose.pose.pose.orientation = Quaternion(*quaternion_from_euler(roll, pitch, yaw))
        self.avg_pose.header.stamp = rospy.Time.now()
        self.avg_pub.publish(self.avg_pose)

        # Hacky way to get the expected MBES ping from avg pose of PF
        # TODO: do this properly :d
        #  (got_result, pf_ping)= self.particles[0].predict_meas(self.avg_pose.pose.pose,
                                                              #  self.beams_real)
        #  if got_result:
            #  self.pf_mbes_pub.publish(pf_ping)


    # TODO: publish markers instead of poses
    #       Optimize this function
    def update_rviz(self):
        self.poses.poses = []
        pose_list = []
        for i in range(self.pc):
            pose_i = Pose()
            pose_i.position.x = self.particles[i].p_pose[0]
            pose_i.position.y = self.particles[i].p_pose[1]
            pose_i.position.z = self.particles[i].p_pose[2]
            pose_i.orientation = Quaternion(*quaternion_from_euler(
                self.particles[i].p_pose[3],
                self.particles[i].p_pose[4],
                self.particles[i].p_pose[5]))

            self.poses.poses.append(pose_i)
            pose_vec = self.particles[i].p_pose
            pose_list.append(pose_vec)
        # Publish particles with time odometry was received
        self.poses.header.stamp = rospy.Time.now()
        self.pf_pub.publish(self.poses)
        self.average_pose(pose_list)


if __name__ == '__main__':

    rospy.init_node('auv_pf')
    try:
        auv_pf()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch pf")
        pass
