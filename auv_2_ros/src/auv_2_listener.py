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
import tf2_ros
import tf_conversions

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2



# import tf

# from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
# from tf.transformations import quaternion_from_euler, euler_from_quaternion

# For pointcloud
# from geometry_msgs.msg import Point

# import geometry_msgs.msg
# import fiducial_msgs.msg
# from fiducial_msgs.msg import FiducialTransformArray, FiducialArray
# from geometry_msgs.msg import Pose, PoseWithCovariance, TransformStamped, PoseStamped 
# from geometry_msgs.msg import Pose, PoseWithCovariance, TransformStamped, PoseStamped 

class AUVListener():

    def __init__(self):
        ## Pull necessary ROS parameters from launch file:
        
        param = rospy.search_param("odometry_topic")
        self.odom_top = rospy.get_param(param)        

        param = rospy.search_param("true_mbes_topic")
        self.true_mbes_top = rospy.get_param(param)

        param = rospy.search_param("simulated_mbes_topic")
        self.sim_mbes_top = rospy.get_param(param)



        # Initialize callback variables


        # Initialize class variables

        # Establish subscription to odom message
        rospy.Subscriber(self.odom_top, Odometry, self.odom_callback)

        # Establish subscription to simulated MBES
        rospy.Subscriber(self.sim_mbes_top, PointCloud2, self.pc_callback)

        # Initialize tf buffer and listener
        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)


        # Establish tf listener for base_link frame (estimated from fiducial_slam)


        # # Initialize tf braodcaster for filtered pose
        # self.br = tf2_ros.TransformBroadcaster()
        # self.t = TransformStamped()
        # self.t.header.frame_id = self.map_frame
        # self.t.child_frame_id = self.pf_frame


        # # Establish subscription to prediction update odometry
        # rospy.Subscriber(self.pred_up_top, Odometry, self.pred_up_callback)
        # # Delay briefly to allow subscribers to find messages
        # rospy.sleep(0.5)

        # # Build the process and observation covariance matrices
        # self.cov_matrices_build()

        # # Initialize array of particle states | # particles x 4 [x, y, theta_z, weight]
        # self.particles = (np.random.rand(self.pc,4)-0.5)*(2*self.init_cov)
        # # Initialize angles on range [-pi, pi]
        # #self.particles[:,2] = (np.random.rand(self.pc,)-0.5)*(2*np.pi)
        # # Initialize all angles to 0 | works better for non-global localization
        # self.particles[:,2] = np.zeros((self.pc,))
        # # Set all particle weights equal
        # self.particles[:,3] = np.ones((self.pc,))

        # # Initialize publisher for estimated pose of vehicle in map frame
        # self.posepub = rospy.Publisher(self.pf_pose_top, PoseWithCovarianceStamped, queue_size=10)
        # self.filt_pose = PoseWithCovarianceStamped()
        # self.filt_pose.header.frame_id = self.map_frame

        # # Initialize publisher for pointcloud
        # self.pointpub = rospy.Publisher(self.pcloud_top, PointCloud, queue_size=10)
        # self.pointcloud = PointCloud()
        # self.pointcloud.header.frame_id = self.map_frame



    # Callback function for odometry message
    def odom_callback(self, odom_msg):
        self.odometry = odom_msg
        #print(self.odometry)
    
    # Callback function for pointclouds
    def pc_callback(self, pc_msg):
        self.sim_mbes = pc_msg
        print(self.sim_mbes)

if __name__ == '__main__':
    # Initialize ROS node
    rospy.init_node('auv_2_listener', anonymous=True)
    rospy.loginfo("Successful initilization of node")
    
    # Create particle filter class
    AUV = AUVListener()
    rospy.loginfo("AUVListener class successfully created")
    
    rospy.spin()
    # Run particle filter
    #pf.run_pf()