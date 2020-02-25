#!/usr/bin/env python 
"""
Module for particle filtering Aruco SLAM for SVEA cars.
Prediction input is Odometry message
    - * Only twist components are used for prediction in this configuration
Observation input is PoseWithCovarianceStamped message
Filtered output is PoseWithCovarianceStamped message
Developed for: KTH Smart Mobility Lab
Developed by: Kyle Coble
""" 
# Standard dependencies
import sys
import os
import math
import rospy
import numpy as np
import tf
import tf2_ros
import tf_conversions

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion

# For pointcloud
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud

#import geometry_msgs.msg
#import fiducial_msgs.msg
#from fiducial_msgs.msg import FiducialTransformArray, FiducialArray
#from geometry_msgs.msg import Pose, PoseWithCovariance, TransformStamped, PoseStamped 
#from geometry_msgs.msg import Pose, PoseWithCovariance, TransformStamped, PoseStamped 

class AUVListener():

    def __init__(self):
        ## Pull necessary ROS parameters from launch file:

        # # Read pose observation topic (pose estimated from aruco detection) 
        # param = rospy.search_param("pose_observation_topic")
        # self.pose_obs_top = rospy.get_param(param)
        # Read pose observation frame id (tf pose estimated from fiducial_slam) 
        param = rospy.search_param("pose_observation_frame")
        self.pose_obs_frame = rospy.get_param(param)        
        # Read map frame id
        param = rospy.search_param("map_frame")
        self.map_frame = rospy.get_param(param)
        # Read filtered pose frame id
        param = rospy.search_param("filtered_pose_frame")
        self.pf_frame = rospy.get_param(param)
        # Read prediction update topic (EKF filtered odometry from IMU and ctrl inputs) 
        param = rospy.search_param("prediction_update_topic")
        self.pred_up_top = rospy.get_param(param)
        # Read particle filter estimate topic (output of this node)) 
        param = rospy.search_param("particle_filtered_pose_topic")
        self.pf_pose_top = rospy.get_param(param)
        # Read point cloud topic (output of this node)) 
        param = rospy.search_param("point_cloud_topic")
        self.pcloud_top = rospy.get_param(param)
        
        # Read particle count 
        param = rospy.search_param("particle_count")
        self.pc = rospy.get_param(param)

        # Read covariance values 
        param = rospy.search_param("initial_estimate_covariance")
        self.init_cov = rospy.get_param(param)
        param = rospy.search_param("linear_process_covariance")
        self.pl_cov = rospy.get_param(param)
        param = rospy.search_param("angular_process_covariance")
        self.pa_cov = rospy.get_param(param)
        param = rospy.search_param("linear_observation_covariance")
        self.ol_cov = rospy.get_param(param)
        param = rospy.search_param("angular_observation_covariance")
        self.oa_cov = rospy.get_param(param)


        # Initialize callback variables
        self.obs_pose = None
        self.pred_odom = None

        # Initialize class variables
        self.time = None
        self.old_time = None
        self.old_theta = 0
        self.ang_z_obs = 0
        self.obs_pose_old = None
        self.innov = np.zeros((self.pc,3))
        self.likeli = np.zeros((self.pc,1))

        # Establish subscription to observation pose
        #rospy.Subscriber(self.pose_obs_top, PoseWithCovarianceStamped, self.obs_pose_callback)
        # Establish tf listener for base_link frame (estimated from fiducial_slam)

        # Initialize listener for estimated pose of markers in map frame
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        # Initialize tf braodcaster for filtered pose
        self.br = tf2_ros.TransformBroadcaster()
        self.t = TransformStamped()
        self.t.header.frame_id = self.map_frame
        self.t.child_frame_id = self.pf_frame


        # Establish subscription to prediction update odometry
        rospy.Subscriber(self.pred_up_top, Odometry, self.pred_up_callback)
        # Delay briefly to allow subscribers to find messages
        rospy.sleep(0.5)

        # Build the process and observation covariance matrices
        self.cov_matrices_build()

        # Initialize array of particle states | # particles x 4 [x, y, theta_z, weight]
        self.particles = (np.random.rand(self.pc,4)-0.5)*(2*self.init_cov)
        # Initialize angles on range [-pi, pi]
        #self.particles[:,2] = (np.random.rand(self.pc,)-0.5)*(2*np.pi)
        # Initialize all angles to 0 | works better for non-global localization
        self.particles[:,2] = np.zeros((self.pc,))
        # Set all particle weights equal
        self.particles[:,3] = np.ones((self.pc,))

        # Initialize publisher for estimated pose of vehicle in map frame
        self.posepub = rospy.Publisher(self.pf_pose_top, PoseWithCovarianceStamped, queue_size=10)
        self.filt_pose = PoseWithCovarianceStamped()
        self.filt_pose.header.frame_id = self.map_frame

        # Initialize publisher for pointcloud
        self.pointpub = rospy.Publisher(self.pcloud_top, PointCloud, queue_size=10)
        self.pointcloud = PointCloud()
        self.pointcloud.header.frame_id = self.map_frame

    ##### Primary particle filter functions #####

    # Callback function for observation pose subscription (from aruco_detect)
    def obs_pose_callback(self, obs_pose_msg):
        self.obs_pose = obs_pose_msg

    # Callback function for prediction odometry subscription (from EKF)
    def pred_up_callback(self, pred_up_msg):
        self.pred_odom = pred_up_msg