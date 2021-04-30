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
from std_msgs.msg import Float32, Header, Bool, Float32MultiArray, ByteMultiArray
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf.transformations import translation_matrix, translation_from_matrix
from tf.transformations import quaternion_matrix, quaternion_from_matrix
from tf.transformations import rotation_matrix, rotation_from_matrix

from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge

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

# For large numbers 
from mpmath import mpf

# train gps simultaneously as rbpf slam is run
# from train_pf_gp import Train_gps

class atree():
    def __init__(self, ID, parent, trajectory, observations):
        self.ID = ID
        self.parent = parent
        self.trajectory = trajectory
        self.observations = observations
        self.children = []

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
        self.n_inducing = rospy.get_param("~n_inducing", 300)
        self.storage_path = rospy.get_param("~data_path") #'/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/data/results/'

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
        self.count_pings = 0
        self.prev_mbes = PointCloud2()
        self.poses = PoseArray()
        self.poses.header.frame_id = odom_frame
        self.avg_pose = PoseWithCovarianceStamped()
        self.avg_pose.header.frame_id = odom_frame
        self.targets = np.zeros((1,))
        self.firstFit = True
        self.one_time = True
        self.time2resample = False
        self.count_training = 0
        self.pw = [1.e-50] * self.pc # Start with almost zero weight
        self.resample_th = 1 / self.pc - 0.1 # when to resample
        # for ancestry tree
        self.observations = np.zeros((1,3)) 
        self.p_ID = 0
        self.tree_list = []

        


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
        # self.recordData2gp()
        train_gp_topic = rospy.get_param('~train_gp_topic', '/training_gps')
        self.gp_pub = rospy.Publisher(train_gp_topic, numpy_msg(Floats), queue_size=10)
        # Subscribe to gp variance and mean
        rospy.Subscriber('/gp_meanvar', numpy_msg(Floats), self.gp_meanvar_cb, queue_size=10)

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
            # self.particles[i].gp = gp.SVGP(self.n_inducing) # doing this in another node to save time
            self.particles[i].ID = self.p_ID
            self.p_ID += 1
        
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
        return mbes_gp, sigma, mu

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
        self.count_pings += 1

    def odom_callback(self, odom_msg):
        self.time = odom_msg.header.stamp.to_sec()
        if self.old_time and self.time > self.old_time:
            # Motion prediction
            self.predict(odom_msg)    
            
            if self.latest_mbes.header.stamp > self.prev_mbes.header.stamp:    # How often to resample, if a new measurement
                # Measurement update if new one received
                # print("Going into update :")
                self.update(self.latest_mbes, odom_msg)
                self.prev_mbes = self.latest_mbes
                
                # Particle resampling
                # self.resample(weights)
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
    
    def trainGP(self):
        if self.firstFit: # Only enter ones
            self.firstFit = False 
            for i in range(0,self.pc):
                inputs = self.particles[i].cloud
                # train each particles gp
                self.particles[i].gp.fit(inputs, self.targets, n_samples= int(self.n_inducing/2), max_iter=int(self.n_inducing/2), learning_rate=1e-1, rtol=1e-4, ntol=100, auto=False, verbose=True)
                # save a plot of the gps
                self.particles[i].gp.plot(inputs, self.targets, self.storage_path + 'particle' + str(i) + 'training' + str(self.count_training) + '.png', n=100, n_contours=100 )
                # save the path to train again
                gp_path = self.storage_path + 'svgp_particle' + str(i) + '.pth'
                self.particles[i].gp.save(gp_path)
                # empty arrays
                self.particles[i].cloud = np.zeros((1,2))
        
        else: # second or more time to retrain gp
            for i in range(0, self.pc):
                gp_path = self.storage_path + 'svgp_particle' + str(i) + '.pth'
                self.particles[i].gp = gp.SVGP.load(self.n_inducing, gp_path)
                inputs = self.particles[i].cloud
                 # train each particles gp
                self.particles[i].gp.fit(inputs, self.targets, n_samples= int(self.n_inducing/2), max_iter=int(self.n_inducing/2), learning_rate=1e-1, rtol=1e-4, ntol=100, auto=False, verbose=True)
                # save a plot of the gps
                self.particles[i].gp.plot(inputs, self.targets, self.storage_path + 'particle' + str(i) + 'training' + str(self.count_training) + '.png', n=100, n_contours=100 )
                # save the path to train again
                gp_path = self.storage_path + 'svgp_particle' + str(i) + '.pth'
                self.particles[i].gp.save(gp_path)
                # empty arrays
                self.particles[i].cloud = np.zeros((1,2))
        
        self.count_training +=1 # to save plots
        # empty array
        self.targets = np.zeros((1,))
        rospy.loginfo('... GP training successful')

    def gp_meanvar_cb(self, msg):
        arr = msg.data
        idx = int(arr[-1]) # Particle ID
        arr = np.delete(arr, -1)
        n = int(arr.shape[0] / 2)
        cloud = arr.reshape(n,2)
        mu_est = cloud[:,0]
        sigma_est = cloud[:,1]
        # for i in range(0, self.pc):
        #     if self.particles[i].ID == p_id: # from resampling, the order can shift, better to look at the id instead of list index
        #         idx = i
        #         if i != p_id: # to keep track of when (if) the resampling changes the order 
        #             print('particle {} have id {}'.format(i, p_id))
        #         break
        self.calculateLikelihood( mu_est, sigma_est, idx, logLkhood=True)


    def calculateLikelihood(self, mu_est, sigma_est, idx, logLkhood):        
        # collect data
        mu_obs = self.particles[idx].mu_list[0]
        sigma_obs = self.particles[idx].sigma_list[0]
        # pop the latesed used
        self.particles[idx].mu_list.pop(0)
        self.particles[idx].sigma_list.pop(0)
        try:
            # divide the equation into subtasks
            mu = mu_est - mu_obs
            sigma = sigma_est + sigma_obs # sigma_est^2 + sigma_obs ^2
            # convert sigma to a matrix
            sigma = np.diag(sigma)

            if logLkhood:
                if self.firstFit:
                    self.firstFit = False # only need to calculate ones
                    self.dim = len(mu_est) # Dimension
                    print('dim ', self.dim)
                    self.norm_const =  self.dim * np.log (2 * np.pi) 
                    print('power to dim ', self.norm_const)

                nom = np.dot(  np.dot( mu , np.linalg.inv(sigma) ) , np.transpose(mu))
                _, detSigma = np.linalg.slogdet(sigma)
                logdetSigma = np.log(detSigma)
                lkhood = -0.5 * (nom + self.norm_const + logdetSigma)
                lkhood = 1/lkhood # since it's negative, the highest negative value should have the lowest weight (e^-inf = 0)
                # print('log likelihood  ', lkhood)
                # lkhood = np.exp(lkhood)

            else:
                if self.firstFit:
                    self.firstFit = False # only need to calculate ones
                    self.dim = len(mu_est) # Dimension
                    print('dim ', self.dim)
                    self.norm_const = np.sqrt( np.power( mpf(2 * np.pi) , self.dim)) # mpf can handle large float numbers
                    print('power to dim ', self.norm_const)
                print('power to dim ')
                print(self.norm_const)

                nom = -0.5 * np.dot(  np.dot( mu , np.linalg.inv(sigma) ) , np.transpose(mu)) # -1/2 * mu^T * Sigma^-1 * mu
                # print('nominator is ', nom)
                # sq1 = np.power( mpf(2 * np.pi) , dim) # mpf can handle large float numbers
                # sq1 = np.exp(dim * np.log(2*np.pi)) # x^y = e^(y ln x)
                _, detSigma = np.linalg.slogdet(sigma)
                # print('sign and det sigma ',sign, detSigma)
                denom = self.norm_const * np.sqrt(detSigma)
                # denom = np.sqrt( np.power( 2 * np.pi, dim) * np.linalg.det(sigma)) # sgrt( (2pi)^dim * Det(sigma))
                # print('denominator is ', denom)
                # calculate the likelihood
                lkhood = np.exp (nom) / denom

        except ValueError:
            print('Likelihood = 0.0')
            lkhood = 0.0

        # print('likelihood of particle ', idx)
        # convert likelihood into weigh
        self.particles[idx].w = lkhood  # particle weight?
        # print(self.particles[idx].w)id
        self.pw[idx] = self.particles[idx].w 

        # when to resample
        if idx == self.pc - 1: # all particles weighted
            self.miss_meas = self.pw.count(0.0)
            weights_array = np.asarray(self.pw)
            # Add small non-zero value to avoid hitting zero
            weights_array += 1.e-200
            self.resample(weights_array)
        
        # for i in range(0, self.pc):
            # gp_cloud = np.load(self.storage_path + 'particle' + str(i) + 'posterior.npy')
            # mu_est = gp_cloud[:,2]
            # sigma_est = gp_cloud[:,3]
        # mu_obs = np.load(self.storage_path + 'particle' + str(idx) + 'mu.npy')
        # sigma_obs = np.load(self.storage_path + 'particle' + str(idx) + 'sigma.npy')

    def update(self, real_mbes, odom):
        # Compute AUV MBES ping ranges in the map frame
        # We only work with z, so we transform them mbes --> map
        #  real_ranges = pcloud2ranges(real_mbes, self.m2o_mat[2,3] + odom.pose.pose.position.z)

        # Beams in real mbes frame
        real_mbes_full = pcloud2ranges_full(real_mbes)
        obs = np.array([[odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z]])
        self.observations = np.append(self.observations, obs, axis=0) # for later comparison
        # print('obs shape ', self.observations.shape)

        # Processing of real pings here
        idx = np.round(np.linspace(0, len(real_mbes_full)-1,
                                           self.beams_num)).astype(int)
        #  real_mbes_ranges = real_ranges[idx]
        real_mbes_full = real_mbes_full[idx]
        # Transform depths from mbes to map frame
        real_mbes_ranges = real_mbes_full[:,2] + self.m2o_mat[2,3] + odom.pose.pose.position.z
        
        # -------------- record target data ------------
        okay = False
        self.targets = np.append(self.targets, real_mbes_ranges, axis=0)
        if self.count_pings % self.record_data == 0: 
            # print('ping count ', self.count_pings)
            okay = True

        # The sensor frame on IGL needs to have the z axis pointing 
        # opposite from the actual sensor direction. However the gp ray tracing
        # needs the opposite.
        # R_flip = rotation_matrix(np.pi, (1,0,0))[0:3, 0:3]
        
        # To transform from base to mbes
        R = self.base2mbes_mat.transpose()[0:3,0:3]

        # Measurement update of each particle
        for i in range(0, self.pc):
            # Compute base_frame from mbes_frame
            p_part, r_mbes = self.particles[i].get_p_mbes_pose()
            r_base = r_mbes.dot(R) # The GP sampling uses the base_link orientation 
                   
            # IGL-based meas model
            exp_mbes = self.draper.project_mbes(np.asarray(p_part), r_mbes,
                                                self.beams_num, self.mbes_angle)
            exp_mbes = exp_mbes[::-1] # Reverse beams for same order as real pings

            # GP meas model
            exp_mbes, exp_sigs, mu_obs = self.gp_meas_model(real_mbes_full, p_part, r_base)
            self.particles[i].meas_cov = np.diag(exp_sigs)
            
            # Publish (for visualization)
            # Transform points to MBES frame (same frame than real pings)
            rot_inv = r_mbes.transpose()
            p_inv = rot_inv.dot(p_part)
            mbes = np.dot(rot_inv, exp_mbes.T)
            mbes = np.subtract(mbes.T, p_inv)
            mbes_pcloud = pack_cloud(self.mbes_frame, mbes)
            
            # ----------- record input data --------
            self.particles[i].cloud = np.append(self.particles[i].cloud, exp_mbes[:,:2], axis=0)  # (n,2)
            self.particles[i].sigma_obs = np.append(self.particles[i].sigma_obs, exp_sigs)
            self.particles[i].mu_obs = np.append(self.particles[i].mu_obs, mu_obs) #exp_mbes[:,2])# mu_obs)
            trajectory = np.array([[self.particles[i].p_pose[0], self.particles[i].p_pose[1], self.particles[i].p_pose[2], self.particles[i].p_pose[3], self.particles[i].p_pose[4], self.particles[i].p_pose[5] ]])
            self.particles[i].trajectory_path = np.append(self.particles[i].trajectory_path, trajectory, axis=0) #save the trajectory for later use
            # print('trajectory shape ', self.particles[i].trajectory_path.shape)
            # saving old x,y poses in a new array to plot later
            # self.particles[i].xy = np.append(self.particles[i].xy , [[self.particles[i].p_pose[0],self.particles[i].p_pose[1]]], axis=0)
            
            #  mbes_pcloud = pack_cloud(self.map_frame, exp_mbes)
            self.pcloud_pub.publish(mbes_pcloud)
            self.particles[i].compute_weight(exp_mbes, real_mbes_ranges)

        if okay: 
            # np.save(self.targets_file, self.targets)
            # print('\nData recorded.\n')
            rospy.loginfo('Training gps... ')
            for i in range(0, self.pc):
                cloud_arr = np.zeros((len(self.targets),3))
                cloud_arr[:,:2] = self.particles[i].cloud
                cloud_arr[:,2] = self.targets
                cloud_arr = cloud_arr.reshape(len(self.targets)*3, 1)
                cloud_arr = np.append(cloud_arr, i) # insert particle index
                for _ in range(3):
                    cloud_arr = np.delete(cloud_arr, 0) # delete the first index of each array, since its not a part of the data
                msg = Floats()
                msg.data = cloud_arr
                # self.trainGP()
                self.gp_pub.publish(msg)

                # save observation data to compare
                self.particles[i].mu_obs = np.delete(self.particles[i].mu_obs, 0)
                self.particles[i].sigma_obs = np.delete(self.particles[i].sigma_obs, 0)
                self.particles[i].mu_list.append(self.particles[i].mu_obs)
                self.particles[i].sigma_list.append(self.particles[i].sigma_obs)

                # fname = self.storage_path + 'particle' + str(i) + 'mu.npy'
                # np.save(fname, self.particles[i].mu_obs)
                # fname = self.storage_path + 'particle' + str(i) + 'sigma.npy' 
                # np.save(fname, self.particles[i].sigma_obs)
                # np.save(self.cloud_file, cloud_arr)
                # np.save(self.pxy_file[i], self.particles[i].xy)

                
                # empty arrays
                self.particles[i].cloud = np.zeros((1,2))
                self.particles[i].mu_obs = np.zeros((1,))
                self.particles[i].sigma_obs = np.zeros((1,))
            self.targets = np.zeros((1,))

        # weights = []
        # print('\n\n NEW WWWW \n\n')
        # for i in range(self.pc):
        #     weights.append(self.particles[i].w) # REMEMBER the new particles need to ärva the old ones gp's.
        #     print(self.particles[i].w)

        # Number of particles that missed some beams 
        # (if too many it would mess up the resampling)
        # self.miss_meas = weights.count(0.0)
        # weights_array = np.asarray(weights)
        # # Add small non-zero value to avoid hitting zero
        # weights_array += 1.e-200

        # return weights_array

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
        weights /= weights.sum()
        # print('should I resample? ', self.time2resample)
        # print(weights)


        N_eff = self.pc
        if weights.sum() == 0.:
            rospy.loginfo("All weights zero!")
        else:
            N_eff = 1/np.sum(np.square(weights))

        self.n_eff_mask.pop(0)
        self.n_eff_mask.append(N_eff)
        self.n_eff_filt = self.moving_average(self.n_eff_mask, 3) 
        # print ("N_eff ", N_eff)
        # print('n_eff_filt ', self.n_eff_filt)
        # print ("Missed meas ", self.miss_meas)
                
        # Resampling?
        # if self.n_eff_filt < self.pc/2. and self.miss_meas < self.pc/2.:
        # if N_eff < self.pc-1: #. and self.miss_meas < self.pc/2.:
        if self.time2resample:
        # if any(t < self.resample_th for t in weights):
            rospy.loginfo('resampling')
            indices = residual_resample(weights)
            keep = list(set(indices))
            lost = [i for i in range(self.pc) if i not in keep]
            dupes = indices[:].tolist()
            for i in keep:
                dupes.remove(i)

            self.ancestry_tree(lost, dupes) # save parent and children
            self.reassign_poses(lost, dupes)
            # check if ancestry tree works
            for p in self.tree_list:
                print('particle {} have parent {} and children {} '.format(p.ID, p.parent, p.children, )) #p.trajectory.shape, p.observations.shape ))
            # Add noise to particles
            for i in range(self.pc):
                self.particles[i].add_noise(self.res_noise_cov)
                # clear data
                self.particles[i].trajectory_path = np.zeros((1,6))
            self.observations = np.zeros((1,3))

        # resample every other time
        self.time2resample = not self.time2resample

    def ancestry_tree(self, lost, dupes):
        if self.one_time:
            self.one_time = False
            for i in range(self.pc):
                particle_tree = atree(self.particles[i].ID, None, self.particles[i].trajectory_path, self.observations )
                self.tree_list.append(particle_tree) # ID = index
        print('how many dupes: ', dupes)
        print('how many lost: ', lost)
        for i in range(len(lost)):
            self.particles[lost[i]].ID = self.p_ID
            # idx_lost = self.particles[lost[i]].ID
            idx_child = self.p_ID
            idx_parent = self.particles[dupes[i]].ID
            particle_tree = atree(idx_child, idx_parent, self.particles[lost[i]].trajectory_path, self.observations )
            self.tree_list.append(particle_tree)
            self.tree_list[idx_parent].children.append(idx_child)
            self.p_ID += 1
            

        # for i in range(self.pc):
        #     particle_tree = atree(self.p_ID, self.particles[dupes[i]].ID, self.particles[lost[i]].trajectory, self.observations )
        #     self.tree_list.append(particle_tree)
        #     self.particles[lost[i]].ID = self.p_ID
        # particle_tree = atree(self.p_ID, self.particles[dupes[i]].ID, self.particles[lost[i]].trajectory, self.observations )
        # self.tree_list.append(particle_tree)
        # self.p_ID += 1

        # merge parent and child if only one child
        # for i in range(self.pc):
        #     if len(self.particles[i].children) == 1:
        #         self.particles[i] = self.particles[i].children[0]

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

    rospy.init_node('rbpf_slam_node', disable_signals=False)
    try:
        rbpf_slam()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch rbpf_node")
        pass
