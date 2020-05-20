#!/usr/bin/env python 

# Standard dependencies
import sys
import os
import math
import rospy
import numpy as np
import tf
import tf2_ros

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion, Transform, TransformStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf.transformations import translation_matrix, translation_from_matrix
from tf.transformations import quaternion_matrix, quaternion_from_matrix

# For sim mbes action client
import actionlib
from auv_2_ros.msg import MbesSimGoal, MbesSimAction
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


class Particle():
    def __init__(self, index, mbes_tf_matrix, meas_cov=0.01, process_cov=[0., 0., 0.], map_frame='map', meas_as='/mbes_server'):
        self.index = index # index starts from 0
        # self.weight = 1.
        self.pose = Pose()
        self.pose.orientation.w = 1.
        self.map_frame = map_frame
        self.mbes_tf_mat = mbes_tf_matrix
        self.meas_cov = meas_cov
        self.process_cov = np.asarray(process_cov)
        self.w = 0.
        self.log_w = 0.

        # Initialize connection to MbesSim action server
        self.ac_mbes = actionlib.SimpleActionClient(meas_as, MbesSimAction)
        rospy.loginfo("Waiting for MbesSim action server ")
        self.ac_mbes.wait_for_server()
        rospy.loginfo("MbesSim action client ")

    def motion_pred(self, odom_t, dt):
        
        xv = odom_t.twist.twist.linear.x
        yv = odom_t.twist.twist.linear.y
        yaw_v = odom_t.twist.twist.angular.z

        """
        There should be a faster way to compute noise_vec
        """
        noise_vec = (np.sqrt(self.process_cov)*np.random.randn(1, 3)).flatten()

        particl_quat = (self.pose.orientation.x,
                        self.pose.orientation.y,
                        self.pose.orientation.z,
                        self.pose.orientation.w)
        _, _, yaw = euler_from_quaternion(particl_quat)

        self.pose.position.x += xv * dt * math.cos(yaw) + noise_vec[0] + yv * dt * math.sin(yaw)
        self.pose.position.y += xv * dt * math.sin(yaw) + noise_vec[1] + yv * dt * math.cos(yaw)
        yaw += yaw_v * dt + noise_vec[2]
        """
        depth, roll, & pitch are known from sensors
        """
        self.pose.position.z = odom_t.pose.pose.position.z
        roll, pitch, _ = euler_from_quaternion([odom_t.pose.pose.orientation.x,
                                               odom_t.pose.pose.orientation.y,
                                               odom_t.pose.pose.orientation.z,
                                               odom_t.pose.pose.orientation.w])

        self.pose.orientation = Quaternion(*quaternion_from_euler(roll, pitch, yaw))

    def meas_update(self, mbes_meas_ranges):
        mbes_i = self.simulate_mbes()
        mbes_i_ranges = pcloud2ranges(mbes_i, self.pose)
        self.w, self.log_w = self.weight(mbes_meas_ranges, mbes_i_ranges)

    def weight(self, mbes_meas_ranges, mbes_sim_ranges ):

        C = len(mbes_meas_ranges)*math.log(math.sqrt(2*math.pi*self.meas_cov))

        try: # Sometimes there is no result for mbes_sim_ranges
            mse = ((mbes_meas_ranges - mbes_sim_ranges)**2).mean()
            """
            Calculate regular weight AND log weight for now
            Decide whether to use log or regular weights
            """
            w = math.exp(-mse/(2*self.meas_cov))
            log_w = C - mse/(2*self.meas_cov)
        except:
            rospy.loginfo('Caught exception in auv_pf.measurement() function')
            log_w = -1.e100 # A very large negative value
            w = 1.e-300 # avoid round-off to zero
        
        return w, log_w
 
    def simulate_mbes(self):

        # Find particle's mbes pose without broadcasting/listening to tf transforms
        particle_tf = Transform()
        particle_tf.translation = self.pose.position
        particle_tf.rotation    = self.pose.orientation
        mat_part = matrix_from_tf(particle_tf)

        trans_mat = np.dot(mat_part, self.mbes_tf_mat)

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
        self.ac_mbes.send_goal(mbes_goal)
        self.ac_mbes.wait_for_result()
        mbes_res = self.ac_mbes.get_result()

        # Pack result into PointCloud2
        mbes_pcloud = PointCloud2()
        mbes_pcloud = mbes_res.sim_mbes
        mbes_pcloud.header.frame_id = self.map_frame

        return mbes_pcloud


    def get_pose_vec(self):
        """
        Returns a list of particle pose elements
        [x, y, z, roll, pitch, yaw]

        :return: List of pose values
        :rtype: List
        """
        pose_vec = []
        pose_vec.append(self.pose.position.x)
        pose_vec.append(self.pose.position.y)
        pose_vec.append(self.pose.position.z)

        quat = (self.pose.orientation.x,
                self.pose.orientation.y,
                self.pose.orientation.z,
                self.pose.orientation.w)
        roll, pitch, yaw = euler_from_quaternion(quat)

        pose_vec.append(roll)
        pose_vec.append(pitch)
        pose_vec.append(yaw)

        return pose_vec

def pcloud2ranges(point_cloud, pose):
        ranges = []
        for p in pc2.read_points(point_cloud, field_names = ("x", "y", "z"), skip_nans=True):
            # starts at left hand side of particle's mbes
            dx = pose.position.x - p[0]
            dy = pose.position.y - p[1]
            dz = pose.position.z - p[2]
            dist = math.sqrt((dx**2 + dy**2 + dz**2))
            ranges.append(dist)
        return np.asarray(ranges)


def matrix_from_tf(transform):
    """
    Converts a geometry_msgs/Transform or 
    geometry_msgs/TransformStamped into a 4x4 
    transformation matrix

    :param transform: Transform from parent->child frame
    :type transform: geometry_msgs/Transform(Stamped)
    :return: Transform as 4x4 matrix
    :rtype: Numpy array (4x4)
    """
    if transform._type == 'geometry_msgs/TransformStamped':
        transform = transform.transform

    trans = (transform.translation.x,
             transform.translation.y,
             transform.translation.z)
    quat_ = (transform.rotation.x,
             transform.rotation.y,
             transform.rotation.z,
             transform.rotation.w)

    tmat = translation_matrix(trans)
    qmat = quaternion_matrix(quat_)
    return np.dot(tmat, qmat)
