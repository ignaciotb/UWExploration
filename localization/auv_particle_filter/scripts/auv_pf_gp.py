#!/usr/bin/env python3

# Standard dependencies
import rospy
import sys
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as rot

from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool
from std_srvs.srv import Empty
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

from tf.transformations import quaternion_from_euler
from tf.transformations import rotation_matrix

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

# For sim mbes action client
from auv_particle import Particle, matrix_from_tf, pcloud2ranges, pack_cloud, pcloud2ranges_full, matrix_from_pose
from resampling import residual_resample, naive_resample, systematic_resample, stratified_resample

# Auvlib
from auvlib.bathy_maps import base_draper
from auvlib.data_tools import csv_data

from scipy.ndimage.filters import gaussian_filter

import time 
import pathlib
import tempfile

class auv_pf(object):

    def __init__(self):
        # Read necessary parameters
        self.pc = rospy.get_param('~particle_count', 10) # Particle Count
        self.map_frame = rospy.get_param('~map_frame', 'map') # map frame_id
        self.mbes_frame = rospy.get_param('~mbes_link', 'mbes_link') # mbes frame_id
        self.base_frame = rospy.get_param('~base_link', 'base_link') # mbes frame_id
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.beams_num = rospy.get_param("~num_beams_sim", 20)
        # self.beams_real = rospy.get_param("~n_beams_mbes", 512)
        self.mbes_angle = rospy.get_param("~mbes_open_angle", np.pi/180. * 60.)
        self.gp_meas_model = rospy.get_param("~gp_meas_model")
        self.gp_torch = rospy.get_param("~gptorch_meas_model")

        # Initialize tf listener
        tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tfBuffer)
        
        # Read covariance values
        meas_std = float(rospy.get_param('~measurement_std', 0.01))
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

        # Global variables
        self.pred_odom = None
        self.n_eff_filt = 0.
        self.n_eff_mask = [self.pc]*3
        self.latest_mbes = PointCloud2()
        self.prev_mbes = PointCloud2()
        self.poses = PoseArray()
        self.poses.header.frame_id = self.odom_frame
        self.avg_pose = PoseWithCovarianceStamped()
        self.avg_pose.header.frame_id = self.odom_frame

        # Initialize particle poses publisher
        pose_array_top = rospy.get_param("~particle_poses_topic", '/particle_poses')
        self.pf_pub = rospy.Publisher(pose_array_top, PoseArray, queue_size=10)

        # Initialize average of poses publisher
        avg_pose_top = rospy.get_param("~average_pose_topic", '/average_pose')
        self.avg_pub = rospy.Publisher(avg_pose_top, PoseWithCovarianceStamped, queue_size=10)

        # Expected meas of PF outcome at every time step
        pf_mbes_top = rospy.get_param("~average_mbes_topic", '/avg_mbes')
        self.pf_mbes_pub = rospy.Publisher(pf_mbes_top, PointCloud2, queue_size=1)

        stats_top = rospy.get_param('~pf_stats_top', 'stats')
        self.stats = rospy.Publisher(stats_top, numpy_msg(Floats), queue_size=10)

        mbes_pc_top = rospy.get_param("~particle_sim_mbes_topic", '/sim_mbes')
        self.pcloud_pub = rospy.Publisher(mbes_pc_top, PointCloud2, queue_size=10)
        
        # Load mesh for raytracing
        if not self.gp_meas_model:
            print("PF loading mesh")

            svp_path = rospy.get_param('~sound_velocity_prof')
            mesh_path = rospy.get_param('~mesh_path')
            
            if svp_path.split('.')[1] != 'cereal':
                sound_speeds = csv_data.csv_asvp_sound_speed.parse_file(svp_path)
            else:
                sound_speeds = csv_data.csv_asvp_sound_speed.read_data(svp_path)  

            data = np.load(mesh_path + "mesh.npz")
            V, F, bounds = data['V'], data['F'], data['bounds'] 

            # Create draper
            self.draper = base_draper.BaseDraper(V, F, bounds, sound_speeds)
            self.draper.set_ray_tracing_enabled(False)            
            data = None
            V = None
            F = None
            bounds = None 
            sound_speeds = None
            print("draper created")
            print("Size of draper: ", sys.getsizeof(self.draper))        
 
        # # Load GP
        if self.gp_meas_model:
            gp_path = rospy.get_param("~gp_path", 'gp.path')
            if self.gp_torch:
                # gpytorch
                from gp_mapping import gp
                rospy.loginfo("Loading GPtorch GP model")
                self.gp = gp.SVGP.load(1000, gp_path + "svgp.pth")
            else:
                # for GPflow GPs
                import tensorflow as tf
                rospy.loginfo("Loading GPflow GP model")
                self.gp = tf.saved_model.load(gp_path + "/svgp")
            
            print("Size of GP: ", sys.getsizeof(self.gp))

        # Subscription to real/sim mbes pings 
        mbes_pings_top = rospy.get_param("~mbes_pings_topic", 'mbes_pings')
        rospy.Subscriber(mbes_pings_top, PointCloud2, self.mbes_cb, queue_size=1)
        
        # Establish subscription to odometry message (intentionally last)
        odom_top = rospy.get_param("~odometry_topic", 'odom')
        rospy.Subscriber(odom_top, Odometry, self.odom_callback, queue_size=100)

        # Create expected MBES beams directions
        angle_step = self.mbes_angle/self.beams_num
        self.beams_dir = []
        for i in range(0, self.beams_num):
            roll_step = rotation_matrix(-self.mbes_angle/2.
                                        + angle_step * i, (1,0,0))[0:3, 0:3]
            rot = roll_step[:,2]
            self.beams_dir.append(rot/np.linalg.norm(rot))
        
        # Shift for fast ray tracing in 2D
        beams = np.asarray(self.beams_dir)
        n_beams = len(self.beams_dir)
        self.beams_dir_2d = np.concatenate((beams[:,0].reshape(n_beams,1), 
                                            np.roll(beams[:, 1:3], 1, 
                                                    axis=1).reshape(n_beams,2)), axis=1)

        self.beams_dir_2d = np.array([1,-1,1])*self.beams_dir_2d

        # Transforms from auv_2_ros
        try:
            rospy.loginfo("Waiting for transforms")
            m2o_tf = tfBuffer.lookup_transform(self.map_frame, self.odom_frame,
                                               rospy.Time(0), rospy.Duration(35))
            self.m2o_mat = matrix_from_tf(m2o_tf)
            rospy.loginfo("Got map to odom")

            mbes_tf = tfBuffer.lookup_transform(self.base_frame, self.mbes_frame,
                                                rospy.Time(0), rospy.Duration(35))
            self.base2mbes_mat = matrix_from_tf(mbes_tf)

            rospy.loginfo("Transforms locked - pf node")
        except:
            rospy.loginfo("ERROR: Could not lookup transform from base_link to mbes_link")

        # Initialize list of particles
        self.particles = np.empty(self.pc, dtype=object)
        for i in range(self.pc):
            self.particles[i] = Particle(self.beams_num, self.pc, i, self.base2mbes_mat,
                                         self.m2o_mat, init_cov=init_cov, meas_std=meas_std,
                                         process_cov=motion_cov)
      
        # Topic to signal end of survey
        finished_top = rospy.get_param("~survey_finished_top", '/survey_finished')
        rospy.Subscriber(finished_top, Bool, self.synch_cb)
        self.survey_finished = False

        # Start timing now
        self.time = rospy.Time.now().to_sec()
        self.old_time = rospy.Time.now().to_sec()

        # Create particle to compute DR
        self.dr_particle = Particle(self.beams_num, self.pc, self.pc+1, self.base2mbes_mat,
                                    self.m2o_mat, init_cov=[0.]*6, meas_std=meas_std,
                                    process_cov=motion_cov)

        # PF filter created. Start auv_2_ros survey playing
        rospy.loginfo("Particle filter class successfully created")

        # Empty service to synch the applications waiting for this node to start
        synch_top = rospy.get_param("~synch_topic", '/pf_synch')
        self.srv_server = rospy.Service(synch_top, Empty, self.empty_srv)

        # Main timer for PF
        self.mission_finished = False
        self.odom_latest = Odometry()
        pf_period = rospy.get_param("~pf_period")
        rospy.Timer(rospy.Duration(pf_period), self.pf_update, oneshot=False)

        # For active localization PF simulation
        self.enable_pf_update = rospy.get_param("~enable_pf_update")
        self.enable_pf_update_topic = rospy.get_param("~enable_pf_update_topic")
        rospy.Subscriber(self.enable_pf_update_topic, Bool, self.enable_updates)

        rospy.spin()

    def enable_updates(self, msg):
        self.enable_pf_update = msg.data

    def empty_srv(self, req):
        rospy.loginfo("PF Ready")
        return None

    def synch_cb(self, finished_msg):
        rospy.loginfo("PF node: Survey finished received") 
        #  rospy.signal_shutdown("Survey finished")

    def gpflow_meas_model(self, real_mbes_all, real_mbes_ranges):
        # Sample GP here
        mu_all, sigma_all = self.gp.predict_y_compiled(
            np.ndarray.tolist(real_mbes_all[:, 0:2]))
        mu_all_array = np.array(mu_all)
        sigma_array = np.array([sigma_all])

        # Compute weights
        weights = []
        for i in range(0, self.pc):
            mbes_gp = np.concatenate((np.asarray(real_mbes_all[i*self.beams_num:(i*self.beams_num)+self.beams_num])[:, 0:2],
                                      mu_all_array[i*self.beams_num:(i*self.beams_num)+self.beams_num]), axis=1)
            # self.particles[i].meas_cov = np.diag(exp_sigs)

            # For visualization
            mbes_pcloud = pack_cloud(self.map_frame, mbes_gp)
            self.pcloud_pub.publish(mbes_pcloud)

            self.particles[i].compute_weight(mbes_gp, real_mbes_ranges)
            weights.append(self.particles[i].w)

        return weights

    def gptorch_meas_model(self, real_mbes_all, real_mbes_ranges):

        mu_all, sigma_all = self.gp.sample(real_mbes_all[:, 0:2])
        mu_all_array = np.array([mu_all])
        sigma_array = np.array([sigma_all])
        
        # Compute weights
        weights = []
        for i in range(0, self.pc):
            mbes_gp = np.concatenate((np.asarray(real_mbes_all[i*self.beams_num:(i*self.beams_num)+self.beams_num])[:, 0:2],
                                      mu_all_array.T[i*self.beams_num:(i*self.beams_num)+self.beams_num]), axis=1)
            self.particles[i].exp_meas_cov = np.diag(sigma_array)

            # For visualization
            mbes_pcloud = pack_cloud(self.map_frame, mbes_gp)
            self.pcloud_pub.publish(mbes_pcloud)

            self.particles[i].compute_weight(mbes_gp, real_mbes_ranges)
            weights.append(self.particles[i].w)

        return weights

    def mbes_cb(self, msg):
        if not self.mission_finished:
            self.latest_mbes = msg

    def pf_update(self, event):
        if not self.mission_finished and self.enable_pf_update:
            if self.latest_mbes.header.stamp > self.prev_mbes.header.stamp:
                # Measurement update if new one received
                start = time.time()
                weights = self.update(self.latest_mbes, self.odom_latest)
                self.prev_mbes = self.latest_mbes
                print(time.time() - start)
                # Particle resampling
                self.resample(weights)

    def odom_callback(self, odom_msg):
        self.time = odom_msg.header.stamp.to_sec()
        self.odom_latest = odom_msg

        if not self.mission_finished:
            if self.old_time and self.time > self.old_time:
                # Motion prediction
                self.predict(odom_msg)    
                
            self.update_rviz()
            self.publish_stats(odom_msg)

        self.old_time = self.time

    def predict(self, odom_t):
        dt = self.time - self.old_time
        for i in range(0, self.pc):
            self.particles[i].motion_pred(odom_t, dt)

        # Predict DR
        self.dr_particle.motion_pred(odom_t, dt)

    
    def update(self, real_mbes, odom):
        # Compute AUV MBES ping ranges in the map frame
        # We only work with z, so we transform them mbes --> map
        #  real_ranges = pcloud2ranges(real_mbes, self.m2o_mat[2,3] + odom.pose.pose.position.z)

        # Beams in real mbes frame
        real_mbes_full = pcloud2ranges_full(real_mbes)

        # Processing of real pings here
        idx = np.round(np.linspace(0, len(real_mbes_full)-1,
                                           self.beams_num)).astype(int)
        #  real_mbes_ranges = real_ranges[idx]
        real_mbes_full = real_mbes_full[idx]
        # Transform depths from mbes to map frame
        real_mbes_ranges = real_mbes_full[:,2] + self.m2o_mat[2,3] + odom.pose.pose.position.z
        
        # The sensor frame on IGL needs to have the z axis pointing 
        # opposite from the actual sensor direction. However the gp ray tracing
        # needs the opposite.
        # R_flip = rotation_matrix(np.pi, (1,0,0))[0:3, 0:3]
        
        # To transform from base to mbes
        R = self.base2mbes_mat.transpose()[0:3,0:3]

        # Measurement update of each particle
        real_mbes_full_all = []
        weights = []
        for i in range(0, self.pc):
            # Current particle pose in the map frame
            p_part, r_mbes = self.particles[i].get_p_mbes_pose()
            
            # For raytracing on mesh meas model
            if not self.gp_meas_model:
                exp_mbes = self.draper.project_mbes(np.asarray(p_part), r_mbes,
                                                    self.beams_num, self.mbes_angle)
                exp_mbes = exp_mbes[::-1] # Reverse beams for same order as real pings

                # For visualization
                mbes_pcloud = pack_cloud(self.map_frame, exp_mbes)
                self.pcloud_pub.publish(mbes_pcloud)

                # Uncertainty of expected meas from raytracing: leave equal to that of real MBES
                self.particles[i].exp_meas_cov = self.particles[i].meas_cov
                # Compute particle weight
                self.particles[i].compute_weight(exp_mbes, real_mbes_ranges)
                weights.append(self.particles[i].w)
            
            # For both GP-based meas models
            if self.gp_meas_model:
                # Compute particle mbes_frame to map frame transform
                r_base = r_mbes.dot(R) # The GP sampling uses the base_link orientation 
                real_mbes = np.dot(r_base, real_mbes_full.T)
                real_mbes = np.add(real_mbes.T, p_part)
                real_mbes_full_all.append(real_mbes)
            
        if self.gp_meas_model:
            real_mbes_full_all = np.vstack(real_mbes_full_all)
            
            # First GP meas model
            if self.gp_torch:
                weights = self.gptorch_meas_model(
                    real_mbes_full_all, real_mbes_ranges)

            # Second GP meas model
            else:
                weights = self.gpflow_meas_model(
                    real_mbes_full_all, real_mbes_ranges)
                    
        # Number of particles that missed some beams 
        # (if too many it would mess up the resampling)
        self.miss_meas = weights.count(0.0)
        weights_array = np.asarray(weights)
        # Add small non-zero value to avoid hitting zero
        weights_array += 1.e-200

        return weights_array
    
    def publish_stats(self, gt_odom):
        # Send statistics for visualization
        p_odom = self.dr_particle.p_pose
        stats = np.array([self.n_eff_filt,
                          self.pc/2.,
                          gt_odom.pose.pose.position.x,
                          gt_odom.pose.pose.position.y,
                          gt_odom.pose.pose.position.z,
                          self.avg_pose.pose.pose.position.x,
                          self.avg_pose.pose.pose.position.y,
                          self.avg_pose.pose.pose.position.z,
                          p_odom[0],
                          p_odom[1],
                          p_odom[2],
                          self.cov[0,0],
                          self.cov[0,1],
                          self.cov[0,2],
                          self.cov[1,1],
                          self.cov[1,2],
                         self.cov[2,2]], dtype=np.float32)

        self.stats.publish(stats) 


    def ping2ranges(self, point_cloud):
        ranges = []
        cnt = 0
        for p in pc2.read_points(point_cloud, 
                                 field_names = ("x", "y", "z"), skip_nans=True):
            ranges.append(np.linalg.norm(p[-2:]))
        
        return np.asarray(ranges)
    
    def moving_average(self, a, n=3) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def resample(self, weights):
        # Normalize weights
        #  print (weights)
        weights /= weights.sum()

        N_eff = self.pc
        if weights.sum() == 0.:
            rospy.loginfo("All weights zero!")
        else:
            N_eff = 1/np.sum(np.square(weights))

        self.n_eff_mask.pop(0)
        self.n_eff_mask.append(N_eff)
        self.n_eff_filt = self.moving_average(self.n_eff_mask, 3) 
        print ("N_eff ", N_eff)
        print ("Missed meas ", self.miss_meas)
                
        # Resampling?
        if self.n_eff_filt < self.pc/2. and self.miss_meas < self.pc/2.:
        #  if N_eff < self.pc/2. and self.miss_meas < self.pc/2.:
            indices = residual_resample(weights)
            keep = list(set(indices))
            lost = [i for i in range(self.pc) if i not in keep]
            dupes = indices[:].tolist()
            for i in keep:
                dupes.remove(i)

            self.reassign_poses(lost, dupes)
            # Add noise to particles
            for i in range(self.pc):
                self.particles[i].add_noise(self.res_noise_cov)


    def reassign_poses(self, lost, dupes):
        for i in range(len(lost)):
            # Faster to do separately than using deepcopy()
            self.particles[lost[i]].p_pose = self.particles[dupes[i]].p_pose
    
    def average_pose(self, pose_list):

        poses_array = np.array(pose_list)
        ave_pose = poses_array.mean(axis = 0)
        self.avg_pose.pose.pose.position.x = ave_pose[0]
        self.avg_pose.pose.pose.position.y = ave_pose[1]
        self.avg_pose.pose.pose.position.z = ave_pose[2]
        roll  = ave_pose[3]
        pitch = ave_pose[4]

        # Wrap up yaw between -pi and pi        
        poses_array[:,5] = [(yaw + np.pi) % (2 * np.pi) - np.pi 
                             for yaw in  poses_array[:,5]]
        yaw = np.mean(poses_array[:,5])
        
        self.avg_pose.pose.pose.orientation = Quaternion(*quaternion_from_euler(roll,
                                                                                pitch,
                                                                                yaw))
        self.avg_pose.header.stamp = rospy.Time.now()
        
        # Calculate covariance
        self.cov = np.zeros((3, 3))
        for i in range(self.pc):
            dx = (poses_array[i, 0:3] - ave_pose[0:3])
            self.cov += np.diag(dx*dx.T) 
            self.cov[0,1] += dx[0]*dx[1] 
            self.cov[0,2] += dx[0]*dx[2] 
            self.cov[1,2] += dx[1]*dx[2] 
        self.cov /= self.pc
        self.cov[1,0] = self.cov[0,1]
        # print(self.cov)

        self.avg_pose.pose.covariance = [0.]*36
        for i in range(3):
            for j in range(3):
                self.avg_pose.pose.covariance[i*3 + j] = self.cov[i,j]
        
        self.avg_pub.publish(self.avg_pose)

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
            pose_list.append(self.particles[i].p_pose)
        
        # Publish particles with time odometry was received
        self.poses.header.stamp = rospy.Time.now()
        self.pf_pub.publish(self.poses)
        self.average_pose(pose_list)


if __name__ == '__main__':

    rospy.init_node('auv_pf', disable_signals=False)
    try:
        auv_pf()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch pf")
        pass
