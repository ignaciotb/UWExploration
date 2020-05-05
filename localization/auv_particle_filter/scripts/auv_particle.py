#!/usr/bin/env python 

# Standard dependencies
import sys
import os
import math
import rospy
import numpy as np
import tf
import tf2_ros
import tf_conversions
import tf2_msgs.msg # Not sure if needed
import scipy.stats # For weights
from copy import deepcopy

from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from geometry_msgs.msg import Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf.transformations import translation_matrix, translation_from_matrix
from tf.transformations import quaternion_matrix, quaternion_from_matrix

# For sim mbes action client
import actionlib
from auv_2_ros.msg import MbesSimGoal, MbesSimAction
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


class Particle():
    def __init__(self, index, map_frame='map'):
        self.index = index # index starts from 1
        self.weight = 1.
        self.pose = Pose()
        self.pose.orientation.w = 1.
        self.map_frame = map_frame


    def pred_update(self, vel_vec, noise_vec, dt):
        quat = (self.pose.orientation.x,
                self.pose.orientation.y,
                self.pose.orientation.z,
                self.pose.orientation.w)
        _, _, yaw = euler_from_quaternion(quat)

        """
        Include 6 DOF motion model 
        """
        self.pose.position.x += vel_vec[0] * dt * math.cos(yaw) + noise_vec[0] + vel_vec[1] * dt * math.sin(yaw)
        self.pose.position.y += vel_vec[0] * dt * math.sin(yaw) + noise_vec[1] + vel_vec[1] * dt * math.cos(yaw)
        yaw += vel_vec[2] * dt + noise_vec[2] # No need for remainder bc quaternion
        self.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, yaw))


    def simulate_mbes(self, mbes_tf_matrix, mbes_ac):

        # Find particle's mbes pose without broadcasting/listening to tf transforms
        particle_trans = (self.pose.position.x,
                          self.pose.position.y,
                          self.pose.position.z)
        particle_quat_ = (self.pose.orientation.x,
                          self.pose.orientation.y,
                          self.pose.orientation.z,
                          self.pose.orientation.w)

        tmat_part = translation_matrix(particle_trans)
        qmat_part = quaternion_matrix(particle_quat_)
        mat_part = np.dot(tmat_part, qmat_part)

        trans_mat = np.dot(mat_part, mbes_tf_matrix)

        trans = TransformStamped()
        trans.transform.translation.x = translation_from_matrix(trans_mat)[0]
        trans.transform.translation.y = translation_from_matrix(trans_mat)[1]
        trans.transform.translation.z = translation_from_matrix(trans_mat)[2]
        trans.transform.rotation = Quaternion(*quaternion_from_matrix(trans_mat))


        # Build MbesSimGoal to send to action server
        mbes_goal = MbesSimGoal()
        mbes_goal.mbes_pose.header.frame_id = self.map_frame
        # mbes_goal.mbes_pose.child_frame_id = self.mbes_frame_id # The particles will be in a child frame to the map
        mbes_goal.mbes_pose.header.stamp = rospy.Time.now()
        mbes_goal.mbes_pose.transform = trans.transform

        # Get result from action server
        mbes_ac.send_goal(mbes_goal)
        # rospy.loginfo("Waiting for MbesSim action Result")
        mbes_ac.wait_for_result()
        # rospy.loginfo("Got MbesSim action Result")
        mbes_res = mbes_ac.get_result()

        # Pack result into PointCloud2
        mbes_pcloud = PointCloud2()
        mbes_pcloud = mbes_res.sim_mbes
        mbes_pcloud.header.frame_id = self.map_frame

        return mbes_pcloud