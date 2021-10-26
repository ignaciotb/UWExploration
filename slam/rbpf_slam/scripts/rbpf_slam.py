#!/usr/bin/env python3

# Standard dependencies
import os
import math

from numpy.core.fromnumeric import shape
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
from std_srvs.srv import Empty
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
from auvlib.data_tools import csv_data

from scipy.ndimage.filters import gaussian_filter

# gpytorch

from rospy.numpy_msg import numpy_msg

# For large numbers 
from mpmath import mpf

from slam_msgs.msg import PlotPosteriorGoal, PlotPosteriorAction
from slam_msgs.msg import SamplePosteriorGoal, SamplePosteriorAction

class atree(): # ancestry tree
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
        self.map_frame = rospy.get_param('~map_frame', 'map') 
        self.mbes_frame = rospy.get_param('~mbes_link', 'mbes_link') 
        self.base_frame = rospy.get_param('~base_link', 'base_link') 
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.beams_num = rospy.get_param("~num_beams_sim", 20)
        self.beams_real = rospy.get_param("~n_beams_mbes", 512)
        self.mbes_angle = rospy.get_param("~mbes_open_angle", np.pi/180. * 60.)
        self.storage_path = rospy.get_param("~result_path")


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
        self.mbes_history = []
        self.latest_mbes = PointCloud2()
        self.count_pings = 0
        self.prev_mbes = PointCloud2()
        self.poses = PoseArray()
        self.poses.header.frame_id = self.odom_frame
        self.avg_pose = PoseWithCovarianceStamped()
        self.avg_pose.header.frame_id = self.odom_frame
        self.targets = np.zeros((1,))
        self.firstFit = True
        self.one_time = True
        self.time2resample = False
        self.count_training = 0
        self.pw = [1.e-50] * self.pc # Start with almost zero weight
        # for ancestry tree
        self.observations = np.zeros((1,3)) 
        self.mapping= np.zeros((1,3)) 
        self.p_ID = 0
        self.tree_list = []
        self.time4regression = False
        self.n_from = 1
        self.ctr = 0

        # Nacho
        self.pings_since_training = 0
        self.map_updates = 0

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

        self.mbes_pc_top = rospy.get_param("~particle_sim_mbes_topic", '/sim_mbes')
        
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

        # Action server for plotting the GP maps
        self.plot_gp_server = rospy.get_param('~plot_gp_server', 'gp_plot_server')
        self.sample_gp_server = rospy.get_param('~sample_gp_server', 'gp_plot_server')

        # Publish to record data
        train_gp_topic = rospy.get_param('~train_gp_topic', '/training_gps')
        self.gp_pub = rospy.Publisher(train_gp_topic, numpy_msg(Floats), queue_size=100)

        # Subscription to real mbes pings 
        mbes_pings_top = rospy.get_param("~mbes_pings_topic", 'mbes_pings')
        rospy.Subscriber(mbes_pings_top, PointCloud2, self.mbes_real_cb, queue_size=100)
        
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

        # Timer for end of mission: finish when no more odom is being received
        self.mission_finished = False
        # self.time_wo_motion = 5.
        # rospy.Timer(rospy.Duration(self.time_wo_motion), self.mission_finished_cb, oneshot=False)
        self.odom_latest = Odometry()
        self.odom_end = Odometry()

        # Transforms from auv_2_ros
        try:
            rospy.loginfo("Waiting for transforms")
            mbes_tf = tfBuffer.lookup_transform(self.base_frame, self.mbes_frame,
                                                rospy.Time(0), rospy.Duration(35))
            self.base2mbes_mat = matrix_from_tf(mbes_tf)

            m2o_tf = tfBuffer.lookup_transform(self.map_frame, self.odom_frame,
                                               rospy.Time(0), rospy.Duration(35))
            self.m2o_mat = matrix_from_tf(m2o_tf)

            rospy.loginfo("Transforms locked - RBFP node")
        except:
            rospy.loginfo("ERROR: Could not lookup transform from base_link to mbes_link")

        # Initialize list of particles
        self.particles = np.empty(self.pc, dtype=object)
        for i in range(self.pc-1):
            self.particles[i] = Particle(self.beams_num, self.pc, i, self.base2mbes_mat,
                                         self.m2o_mat, init_cov=init_cov, meas_std=meas_std,
                                         process_cov=motion_cov)
            self.particles[i].ID = self.p_ID
            self.p_ID += 1
        
        # Create one particle on top of vehicle for tests with very few
        self.particles[i+1] = Particle(self.beams_num, self.pc, i+1, self.base2mbes_mat,
                                         self.m2o_mat, init_cov=[0.]*6, meas_std=meas_std,
                                         process_cov=[0.]*6)
        self.particles[i].ID = self.p_ID
        self.p_ID += 1
        
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

        # For LC detection
        self.lc_detected = False

        # Main timer for RBPF
        rbpf_period = rospy.get_param("~rbpf_period", '/survey_finished')
        rospy.Timer(rospy.Duration(rbpf_period), self.rbpf_update, oneshot=False)

        # Subscription to real mbes pings 
        lc_manual_topic = rospy.get_param("~lc_manual_topic", 'manual_lc')
        rospy.Subscriber(lc_manual_topic, Bool, self.manual_lc, queue_size=1)

        # Empty service to synch the applications waiting for this node to start
        rospy.loginfo("RBPF successfully instantiated")
        synch_top = rospy.get_param("~synch_topic", '/pf_synch')
        self.srv_server = rospy.Service(synch_top, Empty, self.empty_srv)

        rospy.spin()

    def empty_srv(self, req):
        rospy.loginfo("RBPF Ready")
        return None

    def manual_lc(self, lc_msg):
        self.lc_detected = True

    def mission_finished_cb(self, event):
        if self.odom_latest.pose.pose == self.odom_end.pose.pose and not self.mission_finished:
            print("------AUV hasn't moved for self.time_wo_motion seconds: Mission finished!---------")
            # self.mission_finished = True
            # self.lc_detected = True
            # self.plot_gp_maps()
        
        self.odom_end = self.odom_latest

    def synch_cb(self, finished_msg):
        rospy.loginfo("PF node: Survey finished received") 
        # rospy.signal_shutdown("Survey finished")

    def mbes_real_cb(self, msg):
        if not self.mission_finished:
            # Beams in vehicle mbes frame
            real_mbes_full = pcloud2ranges_full(msg)
            # Selecting only self.beams_num of beams in the ping
            idx = np.round(np.linspace(0, len(real_mbes_full)-1,
                                            self.beams_num)).astype(int)
            # Store in pings history
            self.mbes_history.append(real_mbes_full[idx])
            
            # Store latest mbes msg for timing
            self.latest_mbes = msg

            self.count_pings += 1
            for i in range(self.pc):
                self.particles[i].ctr += 1

            self.pings_since_training += 1

    def rbpf_update(self, event):
        if not self.mission_finished:
            if self.latest_mbes.header.stamp > self.prev_mbes.header.stamp:    
            # Measurement update if new one received
                self.update_maps(self.latest_mbes, self.odom_latest)
                self.prev_mbes = self.latest_mbes

                # If potential LC detected
                if(self.lc_detected ):
                    # Recompute weights
                    weights = self.update_particles_weights(self.latest_mbes, self.odom_latest)
                    # Particle resampling
                    self.resample(weights)
                    self.lc_detected = False


    def odom_callback(self, odom_msg):
        self.time = odom_msg.header.stamp.to_sec()
        self.odom_latest = odom_msg
        
        # Flag to finish mission
        # self.time_wo_motion seconds
        if not self.mission_finished:
            if self.old_time and self.time > self.old_time:
                # Motion prediction
                self.predict(odom_msg)    
            
            # Update stats and visual
            self.update_rviz()
            self.publish_stats(self.odom_latest)

        self.old_time = self.time

    def plot_gp_maps(self):
        print("------ Plot final maps --------")
        R = self.base2mbes_mat.transpose()[0:3,0:3]

        for i in range(0, self.pc):
            # ac_plot = actionlib.SimpleActionClient("/particle_" + str(i) + self.plot_gp_server, 
            #                                         PlotPosteriorAction)
            # ac_plot.wait_for_server()

            part_ping_map = []
            for j in range(0, len(self.mbes_history)): 
                # For particle i, get all its trajectory in the map frame
                p_part, r_mbes = self.particles[i].pose_history[j]
                r_base = r_mbes.dot(R) # The GP sampling uses the base_link orientation 

                part_i_ping_map = np.dot(r_base, self.mbes_history[j].T)
                part_ping_map.append(np.add(part_i_ping_map.T, p_part)) 
            # As array
            pings_i = np.asarray(part_ping_map)
            pings_i = np.reshape(pings_i, (-1,3))   

            self.particles[i].gp.plot(pings_i[:, 0:2], pings_i[:, 2],
                self.storage_path + 'gp_result/' + 'particle_' + str(i) 
                + '_training_' + str(self.count_training) + '.png',
                n=100, n_contours=100 )        
                
            # Send to GP particle server
            # mbes_pcloud = pack_cloud(self.map_frame, pings_i)
            # goal = PlotPosteriorGoal(mbes_pcloud)
            # ac_plot.send_goal(goal)
            # ac_plot.wait_for_result()

            print("GP ", i, " posterior plotted ")


    def predict(self, odom_t):
        dt = self.time - self.old_time
        for i in range(0, self.pc):
            self.particles[i].motion_pred(odom_t, dt)
            self.particles[i].update_pose_history()

        # Predict DR
        self.dr_particle.motion_pred(odom_t, dt)


    def update_particles_weights(self, mbes_ping, odom):

        # Latest ping in vehicle mbes frame
        latest_mbes = pcloud2ranges_full(mbes_ping)
        # Selecting only self.beams_num of beams in the ping
        idx = np.round(np.linspace(0, len(latest_mbes)-1,
                                           self.beams_num)).astype(int)
        latest_mbes = latest_mbes[idx]

        # Transform depths from mbes to map frame: we're only going to update the weights
        # based on the absolute depths, which we know well
        latest_mbes_z = latest_mbes[:,2] + self.m2o_mat[2,3] + odom.pose.pose.position.z

        # Calculate expected meas from the particles GP
        R = self.base2mbes_mat.transpose()[0:3,0:3]
        for i in range(0, self.pc):
            # AS for particle i
            # ac_sample = actionlib.SimpleActionClient("/particle_" + str(i) + self.sample_gp_server, 
            #                                         SamplePosteriorAction)
            # ac_sample.wait_for_server()

            # Convert ping from particle MBES to map frame
            p_part, r_mbes = self.particles[i].pose_history[-1]
            r_base = r_mbes.dot(R) # The GP sampling uses the base_link orientation 
            latest_mbes_map = np.dot(r_base, latest_mbes.T)
            latest_mbes_map = np.add(latest_mbes_map.T, p_part)

            # As array
            beams_i = np.asarray(latest_mbes_map)
            beams_i = np.reshape(beams_i, (-1,3))         
            mu, sigma = self.particles[i].gp.sample(np.asarray(beams_i)[:, 0:2])
            
            # mbes_pcloud = pack_cloud(self.map_frame, pings_i)
            # goal = SamplePosteriorGoal(mbes_pcloud)

            # # Send to as and wait
            # ac_sample.send_goal(goal)
            # ac_sample.wait_for_result()
            # result = ac_sample.get_result()  

            # Sample GP with the ping in map frame
            # mu, sigma = result.mu, result.sigma
            
            mu_array = np.array([mu])

            # TODO: use the sigmas from the GP on the weight calculation
            sigma_array = np.array([sigma])

            # Concatenate sampling points x,y with sampled z
            exp_mbes = np.concatenate((np.asarray(latest_mbes_map)[:, 0:2], mu_array.T), axis=1)

            # Compute particles weight
            self.particles[i].compute_weight(exp_mbes, latest_mbes_z)
        
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


    def update_maps(self, real_mbes, odom):
             
        # To transform from base to mbes
        R = self.base2mbes_mat.transpose()[0:3,0:3]
        
        # If time to retrain GP map
        if self.pings_since_training > 50:
            self.map_updates += 1
            for i in range(0, self.pc):           
                # Transform each MBES ping in vehicle frame to the particle trajectory 
                # (result in map frame)
                part_ping_map = []
                for j in range(0, len(self.mbes_history)): 
                    # For particle i, get all its trajectory in the map frame
                    p_part, r_mbes = self.particles[i].pose_history[j]
                    r_base = r_mbes.dot(R) # The GP sampling uses the base_link orientation 

                    part_i_ping_map = np.dot(r_base, self.mbes_history[j].T)
                    part_ping_map.append(np.add(part_i_ping_map.T, p_part)) 
                # As array
                pings_i = np.asarray(part_ping_map)
                pings_i = np.reshape(pings_i, (-1,3))     
                print(pings_i)       
                    
                # Publish (for visualization)
                mbes_pcloud = pack_cloud(self.map_frame, pings_i)

                self.pcloud_pub = rospy.Publisher("/particle_" + str(i) + self.mbes_pc_top, PointCloud2, queue_size=10)
                self.pcloud_pub.publish(mbes_pcloud)

                # Retrain the particle's GP
                print("Training GP ", i)
                # n_samples = a fourth of the total number of beams
                self.particles[i].gp.fit(pings_i[:,0:2], pings_i[:,2], n_samples= 100, 
                                         max_iter=200, learning_rate=1e-1, rtol=1e-4, 
                                         ntol=100, auto=False, verbose=False)
                # # Plot posterior
                # self.particles[i].gp.plot(pings_i[:,0:2], pings_i[:,2], 
                #                           self.storage_path + 'gp_result/' + 'particle_' + str(i) 
                #                           + '_training_' + str(self.count_training) + '.png',
                #                           n=100, n_contours=100 )
                
                print("GP trained ", i)

            # Reset pings for training
            self.count_training += 1
            self.pings_since_training = 0

   
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
        # if self.n_eff_filt < self.pc/2. and self.miss_meas <= self.pc/2.:
        print('n_eff ', N_eff)
        print("Weights ")
        print(weights)
        if N_eff < self.pc/2. and self.miss_meas <= self.pc/2.:
            self.time2resample = False
            self.ctr = 0
            rospy.loginfo('resampling')
            print ("Missed meas ", self.miss_meas)
            indices = residual_resample(weights)
            keep = list(set(indices))
            lost = [i for i in range(self.pc) if i not in keep]
            dupes = indices[:].tolist()
            for i in keep:
                dupes.remove(i)

            self.reassign_poses(lost, dupes)
            
            # Add noise to particles
            # for i in range(self.pc):
            #     self.particles[i].add_noise(self.res_noise_cov)

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
