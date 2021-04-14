#!/usr/bin/env python3

# Standard dependencies
import os
import math
import rospy
import random
import sys
import numpy as np
import tf2_ros
from scipy.spatial.transform import Rotation as rot
from scipy.ndimage import gaussian_filter1d

from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from geometry_msgs.msg import Transform, Quaternion
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
from rbpf_particle import Particle, matrix_from_tf, pcloud2ranges, pack_cloud, pcloud2ranges_full, matrix_from_pose
from resampling import residual_resample, naive_resample, systematic_resample, stratified_resample

# Auvlib
from auvlib.bathy_maps import base_draper
from auvlib.data_tools import csv_data, all_data, std_data

from scipy.ndimage.filters import gaussian_filter

# gpytorch
from bathy_gps import gp
import time 

# train gps simultaneously as rbpf slam is run
# from train_pf_gp import Train_gps

class rbpf_slam(object):

    def __init__(self):
        # Read necessary parameters
        self.pc = rospy.get_param('~particle_count', 10) # Particle Count
        self.map_frame = rospy.get_param('~map_frame', 'map') # map frame_id
        self.mbes_frame = rospy.get_param('~mbes_link', 'mbes_link') # mbes frame_id
        odom_frame = rospy.get_param('~odom_frame', 'odom')
        meas_model_as = rospy.get_param('~mbes_as', '/mbes_sim_server') # map frame_id
        self.beams_num = rospy.get_param("~num_beams_sim", 20)
        self.beams_real = rospy.get_param("~n_beams_mbes", 512)
        self.mbes_angle = rospy.get_param("~mbes_open_angle", np.pi/180. * 60.)
        self.record_data = rospy.get_param("~record_data", 1)

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
        self.count_mbes = 0
        self.prev_mbes = PointCloud2()
        self.poses = PoseArray()
        self.poses.header.frame_id = odom_frame
        self.avg_pose = PoseWithCovarianceStamped()
        self.avg_pose.header.frame_id = odom_frame
        self.targets = np.zeros((1,))
        


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
        # need every particle to create its own gp 
        # using incomming pings instead of loading data 
        gp_path = rospy.get_param("~gp_path", 'gp.path')
        self.gp = gp.SVGP.load(1000, gp_path)
        print("Size of GP: ", sys.getsizeof(self.gp))

        # Action server for MBES pings sim (necessary to be able to use UFO maps as well)
        sim_mbes_as = rospy.get_param('~mbes_sim_as', '/mbes_sim_server')
        self.as_ping = actionlib.SimpleActionServer('/mbes_sim_server', MbesSimAction, 
                                                    execute_cb=self.mbes_as_cb, auto_start=True)

        # Publish to record data
        self.recordData2gp()
        # self.target_pub = rospy.Publisher('/target_top', numpy_msg(Floats), queue_size=10)
        # self.input_pub = rospy.Publisher('/input_top',numpy_msg(Floats), queue_size=10)
        
        # Publish to train gps
        # self.train_gp_pub = rospy.Publisher('/gps2train', PointCloud2, queue_size=1)
        # Subscription to trained gps
        # train_gp_topic = rospy.get_param('~train_gp_topic', '/trained_gps')
        # rospy.Subscriber(train_gp_topic, PointCloud2, self.train_gp_cb)

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
            rospy.loginfo("Waiting for transforms")
            mbes_tf = tfBuffer.lookup_transform('hugin/base_link', 'hugin/mbes_link',
                                                rospy.Time(0), rospy.Duration(35))
            self.base2mbes_mat = matrix_from_tf(mbes_tf)

            m2o_tf = tfBuffer.lookup_transform(self.map_frame, odom_frame,
                                               rospy.Time(0), rospy.Duration(35))
            self.m2o_mat = matrix_from_tf(m2o_tf)

            rospy.loginfo("Transforms locked - pf node")
        except:
            rospy.loginfo("ERROR: Could not lookup transform from base_link to mbes_link")

        # Initialize list of particles
        self.particles = np.empty(self.pc, dtype=object)
        for i in range(self.pc):
            self.particles[i] = Particle(self.beams_num, self.pc, i, self.base2mbes_mat,
                                         self.m2o_mat, init_cov=init_cov, meas_std=meas_std,
                                         process_cov=motion_cov)
            self.particles[i].gp = gp.SVGP.load(1000, gp_path)
        
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
        self.dr_particle = Particle(self.beams_num, self.pc, self.pc+1, self.base2mbes_mat,
                                    self.m2o_mat, init_cov=[0.]*6, meas_std=meas_std,
                                    process_cov=motion_cov)

        rospy.spin()



    def synch_cb(self, finished_msg):
        rospy.loginfo("PF node: Survey finished received") 
        #  rospy.signal_shutdown("Survey finished")


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
        self.count_mbes += 1

    def odom_callback(self, odom_msg):
        self.time = odom_msg.header.stamp.to_sec()
        if self.old_time and self.time > self.old_time:
            # Motion prediction
            self.predict(odom_msg)    
            
            if self.latest_mbes.header.stamp > self.prev_mbes.header.stamp:    # How often to resample, if a new measurement
                # Measurement update if new one received
                # print("Going into update :")
                weights = self.update(self.latest_mbes, odom_msg)
                self.prev_mbes = self.latest_mbes
                
                # Particle resampling
                self.resample(weights)
                self.update_rviz()
                self.publish_stats(odom_msg)

        self.old_time = self.time

    def predict(self, odom_t):
        dt = self.time - self.old_time
        for i in range(0, self.pc):
            self.particles[i].motion_pred(odom_t, dt)

        # Predict DR
        self.dr_particle.motion_pred(odom_t, dt)

    def recordData2gp(self):
        root_folder = '/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/data/'
        dir_name = 'results/'
        # dir_name = ('results_' + str(time.gmtime().tm_year) + '_' + str(time.gmtime().tm_mon) + '_' + str(time.gmtime().tm_mday) + '___'
        #             + str(time.gmtime().tm_hour) + '_' + str(time.gmtime().tm_min) + '_' + str(time.gmtime().tm_sec) + '/')
        if root_folder[-1] != '/':
            dir_name = '/' + dir_name

        storage_path = root_folder + dir_name
        # input_path = storage_path + 'particles/'
        # xy_path = storage_path + 'xy/'
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
        # if not os.path.exists(xy_path):
        #     os.makedirs(xy_path)
        # if not os.path.exists(input_path):
        #     os.makedirs(input_path)
        
        
        self.cloud_file = storage_path + 'ping_cloud.npy'
        # self.inputs_file = [None]*self.pc
        # self.pxy_file = [None]*self.pc

        # for i in range(0, self.pc):
        #     self.inputs_file[i] = input_path + 'inputs_gp_particle' + str(i) + '.npy'
            # self.pxy_file[i] = xy_path + 'xy' + str(i) + '.npy'
    
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
        
        # -------------- record target data ------------
        okay = False
        # new_data[:,2] = real_mbes_ranges
        self.targets = np.append(self.targets, real_mbes_ranges, axis=0)
        if self.count_mbes % self.record_data == 0: 
            okay = True
            # self.targets = np.append(self.targets, real_mbes_ranges, axis=0)
            # print('  saved target  ')
            # print(real_mbes_ranges)
            # print('end target ')
            # targets =PointCloud2(data=real_mbes_ranges)
            # targets = real_mbes_ranges  # (n,) need to be real_mbes_ranges and not real_mbes_full for the mll(...) to work
            # self.target_pub.publish(targets)
            # np.save(self.targets_file, targets)
        
        # The sensor frame on IGL needs to have the z axis pointing 
        # opposite from the actual sensor direction. However the gp ray tracing
        # needs the opposite.
        # R_flip = rotation_matrix(np.pi, (1,0,0))[0:3, 0:3]
        
        # To transform from base to mbes
        R = self.base2mbes_mat.transpose()[0:3,0:3]
        # Choose random particle
        # gp_idx = random.randint(0,self.pc)
        # tao_hyperparam = self.latest_mbes.header.stamp.to_sec() - self.prev_mbes.header.stamp.to_sec()
        # prev_exist = False
        # print("PREV   ", int(self.prev_mbes.header.stamp.to_sec()))
        # if int(self.prev_mbes.header.stamp.to_sec()) != 0:
        #     prev_exist = True
        # print("number of pings sent out is ", self.count_mbes)
        # print("LATEST DATA")
        # print(self.latest_mbes.data)
        # rospy.loginfo("LATEST STAMP ")
        # rospy.loginfo(self.latest_mbes.header.stamp)
        # rospy.loginfo("PREV STAMP ")
        # rospy.loginfo(self.prev_mbes.header.stamp)
        # rospy.loginfo("HYPER !! ")
        # rospy.loginfo(tao_hyperparam)
        # Measurement update of each particle
        for i in range(0, self.pc):
            # Compute base_frame from mbes_frame
            p_part, r_mbes = self.particles[i].get_p_mbes_pose()
            r_base = r_mbes.dot(R) # The GP sampling uses the base_link orientation 
                   
            # IGL-based meas model
            exp_mbes = self.draper.project_mbes(np.asarray(p_part), r_mbes,
                                                self.beams_num, self.mbes_angle)

            # GP meas model
            # exp_mbes, exp_sigs = self.gp_meas_model(real_mbes_full, p_part, r_base)
            # self.particles[i].meas_cov = np.diag(exp_sigs)
            #  print(exp_sigs)
            exp_mbes = exp_mbes[::-1] # Reverse beams for same order as real pings

            # find input and target
            # rospy.loginfo("A B S")
            # rospy.loginfo(abs(real_mbes_ranges))
            
            # Publish (for visualization)
            # Transform points to MBES frame (same frame than real pings)
            rot_inv = r_mbes.transpose()
            p_inv = rot_inv.dot(p_part)
            mbes = np.dot(rot_inv, exp_mbes.T)
            mbes = np.subtract(mbes.T, p_inv)
            mbes_pcloud = pack_cloud(self.mbes_frame, mbes)
            # hej = mbes_pcloud.header.stamp.to_sec

            
            # ----------- record input data --------
            
                # inputs = PointCloud2(data=exp_mbes[:,0:2])
            self.particles[i].cloud = np.append(self.particles[i].cloud, exp_mbes[:,:2], axis=0)  # (n,3)
            # self.particles[i].cloud = np.append(self.particles[i].cloud, exp_mbes, axis=0)  # (n,3)
            # saving old x,y poses in a new array to plot later
            # self.particles[i].xy = np.append(self.particles[i].xy , [[self.particles[i].p_pose[0],self.particles[i].p_pose[1]]], axis=0)
            # if okay:
                # self.particles[i].inputs = np.append(self.particles[i].inputs, exp_mbes[:,0:2], axis=0)  # (n,2)
                # self.particles[i].xy = np.append(self.particles[i].xy , [[self.particles[i].p_pose[0],self.particles[i].p_pose[1]]], axis=0)

                # inputs = exp_mbes[:,0:2] 
                # np.save(self.inputs_file[i], self.particles[i].inputs)
                # np.save(self.pxy_file[i], self.particles[i].xy)
                # self.input_pub.publish(inputs)
                # self.train_gp_pub.publish(self.particles[i], inputs, targets, i)
                # self.particles[i].gp.fit(inputs, targets, n_samples=6000, max_iter=1000, learning_rate=1e-1, rtol=1e-4, ntol=100, auto=False, verbose=True)
                # print("trained particle number ", i)

            # rospy.loginfo("MAYBE MBES??")
            # rospy.loginfo(inputs.shape)
            
            
            #  mbes_pcloud = pack_cloud(self.map_frame, exp_mbes)
            self.pcloud_pub.publish(mbes_pcloud)
            self.particles[i].compute_weight(exp_mbes, real_mbes_ranges)
            # self.particles[i].compute_GP_weight(exp_mbes, real_mbes_ranges)
        
        if okay: #self.count_mbes == 100: #okay:
            # print('Size target:  {} \n Size input: {} \n'.format(self.targets.shape, self.particles[0].inputs.shape))
            # np.save(self.targets_file, self.targets)
            # for i in range(0, 1):
            cloud_arr = np.zeros((len(self.targets),3))
            cloud_arr[:,:2] = self.particles[0].cloud
            cloud_arr[:,2] = self.targets
            np.save(self.cloud_file, cloud_arr)
                # np.save(self.pxy_file[i], self.particles[i].xy)

            print('\nData recorded.\n')
            print('shape cloud ')
            print(cloud_arr.shape)

        weights = []
        for i in range(self.pc):
            weights.append(self.particles[i].w) # REMEMBER the new particles need to ärva the old ones gp's.

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


class GP_mbes:
    # -------------- Global constants ------------------------------------:
    # MBES parameters
    beam_width = math.pi/180.*60.
    nbr_beams = 512
    # ---------------------------------------------------------------------

    def __init__(self, ping, draper):
        self.ping = ping
        self.R = rot.from_euler("zyx", [ping.heading_, ping.pitch_, ping.roll_ ]).as_dcm()

        ext_calib_R = rot.from_euler("zyx", [ping.heading_, ping.pitch_, ping.roll_  + 0.6]).as_dcm()
        ext_calib_t = ping.pos_ + [10, -20, 0]

        # Extract real MBES ping data
        self.depth = draper.project_altimeter(ping.pos_)
        self.mbes_points = np.asarray(ping.beams)
        # Simulate points with extrnal calibration values
        self.sim_points = draper.project_mbes(ext_calib_t, ext_calib_R, self.nbr_beams, self.beam_width)
        # self.sim_points = draper.project_mbes(ping.pos_, self.R, self.nbr_beams, self.beam_width)
        self.mbes_points_auv_frame = '' # transform_ping_points(self.mbes_points)
        self.sim_points_auv_frame = '' # transform_ping_points(self.sim_points)

    def Training_data(self):
        # Beams distributed along seafloor - AUV's y axis
        self.train_x_mbes = torch.Tensor(list(self.mbes_points_auv_frame[:, 1]))
        self.train_x_sim = torch.Tensor(list(self.sim_points_auv_frame[:, 1]))
        # Depth data - AUV's z axis
        self.train_y_mbes = torch.Tensor(list(self.mbes_points_auv_frame[:, 2]))
        self.train_y_sim = torch.Tensor(list(self.sim_points_auv_frame[:, 2]))

    # Returns MBES points in the AUV's reference frame
    # CHECK IF neccecary 
    def transform_ping_points(self, points):
        """
        Returns a ping's MBES points (along the seafloor)
        in the AUV's reference frame

        :param points: MBES ping points to transform [x, y, z]
        :type points: numpy array: shape(N,3)
        :param ping_pose: MBES ping pose from 'auvlib.data_tools.std_data.mbes_ping'
        :type ping_pose: numpy array: shape(3,)
        :param R: AUV 3x3 rotation matrix
        :type R: numpy array: shape(3,3)

        :return: Transformed MBES points [x,y,z]
        :rtype: Numpy array (nbr_pointsx3)
        """

        # 4x4 transformation matrix (tf: auv -> world)
        tf_world_auv = np.eye(4)
        tf_world_auv[:3,:] = np.concatenate((self.R, np.array([[self.ping.pos_[0], self.ping.pos_[1], self.ping.pos_[2]]]).T), axis=1)
        tf_auv_world = np.linalg.inv(tf_world_auv)

        # Multiply 4x4 tf: auv -> world with points to get points in AUV frame
        temp_points = np.transpose(np.hstack((points,np.ones((points.shape[0],1)))))
        points_auv_frame = np.transpose(np.dot(tf_auv_world, temp_points))[:,:3]

        return points_auv_frame
    
    # Run regression (all lumped into one function for now)
    def run_gpytorch_regression(train_x, train_y): #, training_point_count, max_training_iter_count, noise_threshold, loss_threshold):
        """
        Run regression (all lumped into one function for now)
        """
        # %matplotlib inline
        # %load_ext autoreload
        # %autoreload 2

        # initialize likelihood and model
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = ExactGPModel(train_x, train_y, likelihood)

        # this is for running the notebook in our testing framework
        import os
        smoke_test = ('CI' in os.environ)
        max_train_iter = 2 if smoke_test else max_training_iter_count

        # Find optimal model hyperparameters
        model.train()
        likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        i = 0 # init iteration count
        loss_val = 999. # init loss value for convergence w/ large value

        # Train until convergence
        # while loss_val >= loss_threshold:
        while model.likelihood.noise.item() >= noise_threshold:
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(train_x)
            # Calc loss and backprop gradients
            loss = -mll(output, train_y)
            loss.backward()
            if (i+1) % 10 == 0:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.4f' % (
                    i + 1, max_train_iter, loss.item(),
                    model.covar_module.base_kernel.lengthscale.item(),
                    model.likelihood.noise.item()
                ))
            optimizer.step()

            # Update iteration / convergence tracking
            loss_val = loss.item()
            i += 1
            if i > max_train_iter:
                break

        # Get into evaluation (predictive posterior) mode
        model.eval()
        likelihood.eval()

        # Test points are regularly spaced on range of y-axis
        # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():  # To use LOVE 
            test_x = torch.linspace(min(train_x), max(train_x), training_point_count)
            observed_pred = likelihood(model(test_x))

        return test_x, observed_pred


if __name__ == '__main__':

    rospy.init_node('rbpf_slam_node', disable_signals=False)
    try:
        rbpf_slam()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch rbpf_node")
        pass
