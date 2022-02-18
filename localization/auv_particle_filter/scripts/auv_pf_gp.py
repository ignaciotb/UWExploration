#!/usr/bin/env python3

# Standard dependencies
import math
import rospy
import sys
import numpy as np
import tf2_ros
# import tf
from scipy.spatial.transform import Rotation as rot
from scipy.ndimage import gaussian_filter1d

from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from geometry_msgs.msg import Transform, Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Header, Bool
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf.transformations import translation_matrix, translation_from_matrix
from tf.transformations import quaternion_matrix, quaternion_from_matrix
from tf.transformations import rotation_matrix, rotation_from_matrix

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

# For sim mbes action client
import actionlib
from auv_2_ros.msg import MbesSimGoal, MbesSimAction, MbesSimResult
from auv_particle import Particle, matrix_from_tf, pcloud2ranges, pack_cloud, pcloud2ranges_full, matrix_from_pose
from resampling import residual_resample, naive_resample, systematic_resample, stratified_resample

# Auvlib
from auvlib.bathy_maps import base_draper
from auvlib.data_tools import csv_data, all_data, std_data

from scipy.ndimage.filters import gaussian_filter

# gpytorch
from bathy_gps import gp
import time

class auv_pf(object):

    def __init__(self):
        # Read necessary parameters
        self.namespace = rospy.get_param('~namespace')
        self.pc = rospy.get_param('~particle_count', 10) # Particle Count
        self.map_frame = rospy.get_param('~map_frame', 'map') # map frame_id
        self.mbes_frame = rospy.get_param('~mbes_link',
                                          self.namespace + '/enu/mbes_link')
        odom_frame = rospy.get_param('~odom_frame',
                                     self.namespace + '/odom')
        meas_model_as = rospy.get_param('~mbes_as', '/mbes_sim_server') # map frame_id
        self.beams_num = rospy.get_param("~num_beams_sim", 20)
        self.beams_real = rospy.get_param("~n_beams_mbes", 512)
        self.mbes_angle = rospy.get_param("~mbes_open_angle", np.pi/180. * 60.)

        # Initialize tf listener
        tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tfBuffer)
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

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
        # Subscription to real mbes pings
        mbes_pings_top = rospy.get_param("~mbes_pings_topic", 'mbes_pings')
        rospy.Subscriber(mbes_pings_top, PointCloud2, self.mbes_real_cb)

        # Establish subscription to odometry message (intentionally last)
        odom_top = rospy.get_param("~odometry_topic", 'odom')
        rospy.Subscriber(odom_top, Odometry, self.odom_callback)
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

        # Expected meas of PF outcome at every time step
        pf_mbes_top = rospy.get_param("~average_mbes_topic", '/avg_mbes')
        self.pf_mbes_pub = rospy.Publisher(pf_mbes_top, PointCloud2, queue_size=1)

        stats_top = rospy.get_param('~pf_stats_top', 'stats')
        self.stats = rospy.Publisher(stats_top, numpy_msg(Floats), queue_size=10)

        mbes_pc_top = rospy.get_param("~particle_sim_mbes_topic", '/sim_mbes')
        self.pcloud_pub = rospy.Publisher(mbes_pc_top, PointCloud2, queue_size=10)

        # Load mesh
        svp_path = rospy.get_param('~sound_velocity_prof')
        mesh_path = rospy.get_param('~mesh_path')

        if svp_path.split('.')[1] != 'cereal':
            sound_speeds = csv_data.csv_asvp_sound_speed.parse_file(svp_path)
        else:
            sound_speeds = csv_data.csv_asvp_sound_speed.read_data(svp_path)

        data = np.load(mesh_path)
        V, F, bounds = data['V'], data['F'], data['bounds']
        print("Mesh loaded")

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

        # Load GP
        # gp_path = rospy.get_param("~gp_path", 'gp.path')
        # self.gp = gp.SVGP.load(1000, gp_path)
        # print("Size of GP: ", sys.getsizeof(self.gp))

        # Action server for MBES pings sim (necessary to be able to use UFO maps as well)
        sim_mbes_as = rospy.get_param('~mbes_sim_as', '/mbes_sim_server')
        self.as_ping = actionlib.SimpleActionServer('/mbes_sim_server', MbesSimAction,
                                                    execute_cb=self.mbes_as_cb, auto_start=True)

        # Subscription to real mbes pings
        mbes_pings_top = rospy.get_param("~mbes_pings_topic", 'mbes_pings')
        rospy.Subscriber(mbes_pings_top, PointCloud2, self.mbes_real_cb)

        # Establish subscription to odometry message (intentionally last)
        odom_top = rospy.get_param("~odometry_topic", 'odom')
        rospy.Subscriber(odom_top, Odometry, self.odom_callback)

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

        # Start to play survey data. Necessary to keep the PF and auv_2_ros in synch
        synch_top = rospy.get_param("~synch_topic", '/pf_synch')
        self.synch_pub = rospy.Publisher(synch_top, Bool, queue_size=10)
        msg = Bool()
        msg.data = True

        # Transforms from auv_2_ros
        try:
            rospy.loginfo("Waiting for transform base_link -> mbes_link")
            mbes_tf = tfBuffer.lookup_transform(self.namespace + '/base_link',
                                                self.namespace + '/enu/mbes_link',
                                                # rospy.Time(), rospy.Duration(35))
                                                rospy.Time())
            self.base2mbes_mat = matrix_from_tf(mbes_tf)
            rospy.loginfo("Transform base_link -> mbes_link locked - pf node")
        except:
            rospy.logerr("Could not lookup transform from base_link -> mbes_link")

        try:
            rospy.loginfo("Waiting for transform %s -> %s", self.map_frame, odom_frame)
            m2o_tf = tfBuffer.lookup_transform(self.map_frame, odom_frame,
                                               # rospy.Time(), rospy.Duration(35))
                                               rospy.Time())
            self.m2o_mat = matrix_from_tf(m2o_tf)
            rospy.loginfo("Transform %s -> %s locked - pf node", self.map_frame, odom_frame)
        except:
            rospy.logwarn("Could not lookup transform %s -> %s, setting to zero", self.map_frame, odom_frame)
            m2o_tf = TransformStamped()
            m2o_tf.transform.rotation.w = 1.0
            m2o_tf.header.stamp = rospy.Time.now()
            m2o_tf.header.frame_id = self.map_frame
            m2o_tf.child_frame_id = odom_frame
            self.tf_broadcaster.sendTransform(m2o_tf)
            self.m2o_mat = matrix_from_tf(m2o_tf)

        try:
            rospy.loginfo("Waiting for transform %s -> base_link", self.map_frame)
            base_tf = tfBuffer.lookup_transform(self.map_frame,
                                                self.namespace + '/base_link',
                                                rospy.Time(), rospy.Duration(1.0))
                                                # rospy.Time())
            self.map2base_mat = matrix_from_tf(base_tf)
            rospy.loginfo("Transform %s -> base_link locked - pf node", self.map_frame)
        except:
            rospy.logerr("Could not lookup transform from %s -> base_link", self.map_frame)

        # Initialize list of particles
        # self.particles = np.empty(self.pc, dtype=object)
        # for i in range(self.pc):
            # self.particles[i] = Particle(self.beams_num, self.pc, i, self.base2mbes_mat,
                                         # self.m2o_mat, init_cov=init_cov, meas_std=meas_std,
                                         # process_cov=motion_cov)
        self.particles = np.empty(self.pc, dtype=object)
        for i in range(self.pc):
            self.particles[i] = Particle(self.beams_num, self.pc, i, self.base2mbes_mat,
                                         self.map2base_mat, self.m2o_mat,
                                         init_cov=init_cov, meas_std=meas_std,
                                         process_cov=motion_cov)
        # PF filter created. Start auv_2_ros survey playing
        rospy.loginfo("Particle filter class successfully created")
        self.synch_pub.publish(msg)

        finished_top = rospy.get_param("~survey_finished_top", '/survey_finished')
        self.finished_sub = rospy.Subscriber(finished_top, Bool, self.synch_cb)
        self.survey_finished = False

        # Start timing now
        self.time = rospy.Time.now().to_sec()
        self.old_time = rospy.Time.now().to_sec()

        # Create particle to compute DR
        # self.dr_particle = Particle(self.beams_num, self.pc, self.pc+1, self.base2mbes_mat,
                                    # self.m2o_mat, init_cov=[0.]*6, meas_std=meas_std,
                                    # process_cov=motion_cov)
        self.dr_particle = Particle(self.beams_num, self.pc, self.pc+1, self.base2mbes_mat,
                                    self.map2base_mat, self.m2o_mat, init_cov=[0.]*6,
                                    meas_std=meas_std, process_cov=motion_cov)

        rospy.spin()

    def synch_cb(self, finished_msg):
        rospy.loginfo("PF node: Survey finished received")
        #  rospy.signal_shutdown("Survey finished")

    def gp_sampling(self, p, R):
        h = 100. # Depth of the field of view
        b = h / np.cos(self.mbes_angle/2.)
        n = 80  # Number of sampling points

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

        #  start = time.time()
        p2s = np.full((n, len(p2)), p2)
        direcs = np.full((n, len(direc)), direc)
        i_ranges = np.full((1,n), i_range)*(np.asarray(range(0,n)))
        sampling_points = p2s + i_ranges.reshape(n, 1)*direcs

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
        gp_samples = np.dot(rot_inv, gp_samples.T)
        gp_samples = np.subtract(gp_samples.T, p_inv)

        # Without for loop version
        start = time.time()
        n = len(gp_samples)
        M = len(self.beams_dir)
        v1 = -np.roll(gp_samples, 1, axis=0)
        v1_vec = np.tile(v1, (M,1))
        v2_vec = np.tile(gp_samples + v1, (M,1))
        v3_vec = np.repeat(self.beams_dir_2d, len(gp_samples), axis=0)

        inner23_vec = np.einsum('ij, ij->i', v2_vec, v3_vec)
        t2_vec = np.einsum('ij, ij->i', v1_vec, v3_vec)/inner23_vec
        t2_vec = t2_vec.reshape(M, n)

        hits_vec = np.asarray(np.where((t2_vec[0]>=0) & (t2_vec[0]<=1))[0][1]).reshape(1,1)
        for m in range(1, M):
            new = np.asarray(np.where((t2_vec[m]>=0) & (t2_vec[m] <= 1))[0][1]).reshape(1,1)
            hits_vec = np.concatenate((hits_vec, new), 0)

        # TODO: optimize
        #  rows, cols = np.where((t2_vec[0]>=0) & (t2_vec[0]<=1))

        t1_vec = (np.cross(v2_vec, v1_vec)/inner23_vec.reshape(M*len(gp_samples),1))
        # TODO: This one can be faster too
        t1_vec = np.sqrt(np.sum(t1_vec**2,axis=-1))
        t1_vec = t1_vec.reshape(M, n)

        exp_meas = []
        for m in range(M):
            if len(hits_vec[m]) > 0:
                if t1_vec[m, hits_vec[m]] > 0:
                    exp_meas.append(self.beams_dir[m] * t1_vec[m, hits_vec[m]])

        #  print("Duration mat", time.time() - start)
        #  print("----")

        # Transform back to map frame
        mbes_gp = np.asarray(exp_meas)
        mbes_gp = np.dot(r_mbes, mbes_gp.T)
        mbes_gp = np.add(mbes_gp.T, p)

        return mbes_gp

    def gp_meas_model(self, real_mbes, p_part, r_base):
        # Transform beams to particle mbes frame and compute map coordinates
        real_mbes = np.dot(r_base, real_mbes.T)
        real_mbes = np.add(real_mbes.T, p_part)

        # Sample GP here
        mu, sigma = self.gp.sample(np.asarray(real_mbes)[:, 0:2])
        mu_array = np.array([mu])
        sigma_array = np.array([sigma])

        # Concatenate sampling points x,y with sampled z
        mbes_gp = np.concatenate((np.asarray(real_mbes)[:, 0:2],
                                  mu_array.T), axis=1)
        #  print(sigma)
        return mbes_gp, sigma

    def mbes_real_cb(self, msg):
        self.latest_mbes = msg

    # Action server to simulate MBES for the sim AUV
    def mbes_as_cb(self, goal):

        # Unpack goal
        p_mbes = [goal.mbes_pose.transform.translation.x,
                 goal.mbes_pose.transform.translation.y,
                 goal.mbes_pose.transform.translation.z]
        r_mbes = quaternion_matrix([goal.mbes_pose.transform.rotation.x,
                                    goal.mbes_pose.transform.rotation.y,
                                    goal.mbes_pose.transform.rotation.z,
                                    goal.mbes_pose.transform.rotation.w])[0:3, 0:3]

        # IGL sim ping
        # The sensor frame on IGL needs to have the z axis pointing opposite from the actual sensor direction
        R_flip = rotation_matrix(np.pi, (1,0,0))[0:3, 0:3]
        mbes = self.draper.project_mbes(np.asarray(p_mbes), r_mbes,
                                        goal.beams_num.data, self.mbes_angle)

        mbes = mbes[::-1] # Reverse beams for same order as real pings

        # Transform points to MBES frame (same frame than real pings)
        rot_inv = r_mbes.transpose()
        p_inv = rot_inv.dot(p_mbes)
        mbes = np.dot(rot_inv, mbes.T)
        mbes = np.subtract(mbes.T, p_inv)

        # Pack result
        mbes_cloud = pack_cloud(self.mbes_frame, mbes)
        result = MbesSimResult()
        result.sim_mbes = mbes_cloud
        self.as_ping.set_succeeded(result)

        self.latest_mbes = result.sim_mbes

    def odom_callback(self, odom_msg):
        self.time = odom_msg.header.stamp.to_sec()
        if self.old_time and self.time > self.old_time:
            # Motion prediction
            self.predict_pos(odom_msg)


            if self.latest_mbes.header.stamp > self.prev_mbes.header.stamp:
                # Measurement update if new one received
                weights = self.update(self.latest_mbes, odom_msg)
                self.prev_mbes = self.latest_mbes

                # Particle resampling
                self.resample(weights)
                self.update_rviz()
                self.publish_stats(odom_msg)

        self.old_time = self.time

    def predict_pos(self, odom_t):
        for i in range(0, self.pc):
            self.particles[i].motion_pred_pos(odom_t)
        self.dr_particle.motion_pred_pos(odom_t)

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
        R_flip = rotation_matrix(np.pi, (1,0,0))[0:3, 0:3]

        # To transform from base to mbes
        R = self.base2mbes_mat.transpose()[0:3,0:3]
        # Measurement update of each particle
        for i in range(0, self.pc):
            # Compute base_frame from mbes_frame
            p_part, r_mbes = self.particles[i].get_p_mbes_pose();
            r_base = r_mbes.dot(R) # The GP sampling uses the base_link orientation

            # First GP meas model
            # exp_mbes, exp_sigs = self.gp_meas_model(real_mbes_full, p_part, r_base)
            #  self.particles[i].meas_cov = np.diag(exp_sigs)
            #  print(exp_sigs)

            # Second GP meas model
            #  gp_samples = self.gp_sampling(p_part, r_base)
            #  exp_mbes = self.gp_ray_tracing(r_mbes.dot(R_flip), p_part,
                                           #  gp_samples, self.beams_num)

            # IGL-based meas model
            exp_mbes = self.draper.project_mbes(np.asarray(p_part), r_mbes,
                                                 self.beams_num, self.mbes_angle)
            #TODO:(aldoteran)
            exp_mbes = exp_mbes[::-1] # Reverse beams for same order as real pings

            # Publish (for visualization)
            # Transform points to MBES frame (same frame than real pings)
            rot_inv = r_mbes.transpose()
            p_inv = rot_inv.dot(p_part)
            mbes = np.dot(rot_inv, exp_mbes.T)
            mbes = np.subtract(mbes.T, p_inv)
            mbes_pcloud = pack_cloud(self.namespace + "/enu/mbes_link", mbes)

            #  mbes_pcloud = pack_cloud(self.map_frame, exp_mbes)
            self.pcloud_pub.publish(mbes_pcloud)

            self.particles[i].compute_weight(exp_mbes, real_mbes_ranges)

        weights = []
        for i in range(self.pc):
            weights.append(self.particles[i].w)

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
        self.avg_pub.publish(self.avg_pose)

        # Calculate covariance
        self.cov = np.zeros((3, 3))
        for i in range(self.pc):
            dx = (poses_array[i, 0:3] - ave_pose[0:3])
            self.cov += np.diag(dx*dx.T)
            self.cov[0,1] += dx[0]*dx[1]
            self.cov[0,2] += dx[0]*dx[2]
            self.cov[1,2] += dx[1]*dx[2]
        self.cov /= self.pc

        # TODO: exp meas from average pose of the PF, for change detection


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
