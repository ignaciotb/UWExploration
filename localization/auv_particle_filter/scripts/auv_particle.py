#!/usr/bin/env python

# Standard dependencies
import sys
import os
import math
import rospy
import numpy as np
import tf
import tf2_ros
from scipy.stats import multivariate_normal

from geometry_msgs.msg import Pose
from geometry_msgs.msg import Quaternion, Transform, TransformStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf.transformations import translation_matrix, translation_from_matrix
from tf.transformations import quaternion_matrix, quaternion_from_matrix

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped, Pose

from auv_2_ros.msg import MbesSimGoal

class Particle(object):
    def __init__(self, index, p_num, mbes_tf_matrix, m2o_matrix, init_cov=[0.,0.,0.,0.,0.,0.],
                 meas_cov=0.01, process_cov=[0.,0.,0.,0.,0.,0.], map_frame='map', odom_frame='odom', pc_mbes_top='/sim_mbes'):

        self.p_num = p_num # index starts from 0
        self.index = index
        # self.weight = 1.
        self.p_pose = Pose()
        self.odom_frame = odom_frame
        self.map_frame = map_frame
        self.mbes_tf_mat = mbes_tf_matrix
        self.m2o_tf_mat = m2o_matrix
        self.init_cov = init_cov
        self.meas_cov = meas_cov
        self.process_cov = np.asarray(process_cov)
        self.w = 0.
        self.log_w = 0.


        self.pcloud_pub = rospy.Publisher(pc_mbes_top, PointCloud2, queue_size=10)
        self.pose_pub = rospy.Publisher('/pf/particles', PoseStamped, queue_size=10)

        # Distribute particles around initial pose
        self.p_pose.position.x = 0.
        self.p_pose.position.y = 0.
        self.p_pose.position.z = 0.
        self.p_pose.orientation.x = 0.
        self.p_pose.orientation.y = 0.
        self.p_pose.orientation.z = 0.
        self.p_pose.orientation.w = 1.

        self.add_noise(init_cov)

    def add_noise(self, noise):
        noise_cov =np.diag(noise)
        roll, pitch, yaw = euler_from_quaternion([self.p_pose.orientation.x,
                                          self.p_pose.orientation.y,
                                          self.p_pose.orientation.z,
                                          self.p_pose.orientation.w])

        current_pose = np.array([self.p_pose.position.x,
                                self.p_pose.position.y,
                                self.p_pose.position.z,
                                roll,
                                pitch,
                                yaw])[np.newaxis]

        noisy_pose = current_pose.T + np.matmul(np.sqrt(noise_cov), np.random.randn(6,1))

        self.p_pose.position.x = noisy_pose[0][0]
        self.p_pose.position.y = noisy_pose[1][0]
        self.p_pose.position.z = noisy_pose[2][0]
        self.p_pose.orientation = Quaternion(*quaternion_from_euler(noisy_pose[3][0],
                                                            noisy_pose[4][0],
                                                            noisy_pose[5][0]))


    def fullRotation(self, roll, pitch, yaw):
        rot_z = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],
                          [np.sin(yaw), np.cos(yaw), 0.0],
                          [0., 0., 1]])
        rot_y = np.array([[np.cos(pitch), 0.0, np.sin(pitch)],
                          [0., 1., 0.],
                          [-np.sin(pitch), np.cos(pitch), 0.0]])
        rot_x = np.array([[1., 0., 0.],
                          [0., np.cos(roll), -np.sin(roll)],
                          [0., np.sin(roll), np.cos(roll)]])

        rot_t = np.matmul(rot_z, np.matmul(rot_y, rot_x))
        return rot_t


    def motion_pred(self, odom_t, dt):
        # Generate noise
        noise_vec = (np.sqrt(self.process_cov)*np.random.randn(1, 6)).flatten()

        # Angular motion
        [roll, pitch, yaw] = euler_from_quaternion([self.p_pose.orientation.x,
                                            self.p_pose.orientation.y,
                                            self.p_pose.orientation.z,
                                            self.p_pose.orientation.w])

        vel_rot = np.array([odom_t.twist.twist.angular.x,
                            odom_t.twist.twist.angular.y,
                            odom_t.twist.twist.angular.z])

        rot_t = np.array([roll, pitch, yaw]) + vel_rot * dt + noise_vec[3:6]

        roll_t = ((rot_t[0]) + 2 * np.pi) % (2 * np.pi)
        pitch_t = ((rot_t[1]) + 2 * np.pi) % (2 * np.pi)
        yaw_t = ((rot_t[2]) + 2 * np.pi) % (2 * np.pi)

        self.p_pose.orientation = Quaternion(*quaternion_from_euler(roll_t,
                                                                    pitch_t,
                                                                    yaw_t))

        # Linear motion
        vel_p = np.array([odom_t.twist.twist.linear.x,
                         odom_t.twist.twist.linear.y,
                         odom_t.twist.twist.linear.z])

        rot_mat_t = self.fullRotation(roll_t, pitch_t, yaw_t)
        step_t = np.matmul(rot_mat_t, vel_p * dt) + noise_vec[0:3]

        self.p_pose.position.x += step_t[0]
        self.p_pose.position.y += step_t[1]
        self.p_pose.position.z += step_t[2]

    def meas_update(self, mbes_meas_ranges):
        # Predict mbes ping given current particle pose and map
        mbes_goal = self.predict_meas()
        return mbes_goal

    def update_weight(self, mbes_pc, mbes_meas_ranges):
        mbes_i_ranges = pcloud2ranges(mbes_pc, self.p_pose)
        #  print "Particle ", self.index
        #  print(mbes_meas_ranges)
        #  print(mbes_i_ranges)

        # Publish (for visualization)
        self.pcloud_pub.publish(mbes_pc)

        # Update particle weights
        #  self.w = self.weight_mv(mbes_meas_ranges, mbes_i_ranges)
        #  print "MV ", self.w
        #  self.w = self.weight_avg(mbes_meas_ranges, mbes_i_ranges)
        #  print "Avg ", self.w
        self.w = self.weight_grad(mbes_meas_ranges, mbes_i_ranges)
        #  print "Gradient", self.w
        return

    def weight_grad(self, mbes_meas_ranges, mbes_sim_ranges ):
        if len(mbes_meas_ranges) == len(mbes_sim_ranges):
            grad_meas = np.gradient(mbes_meas_ranges)
            grad_expected = np.gradient(mbes_sim_ranges)
            w_i = multivariate_normal.pdf(grad_expected, mean=grad_meas, cov=self.meas_cov)
        else:
            print ("missing pings!")
            w_i = 0.
        return w_i


    def weight_mv(self, mbes_meas_ranges, mbes_sim_ranges ):
        if len(mbes_meas_ranges) == len(mbes_sim_ranges):
            w_i = multivariate_normal.pdf(mbes_sim_ranges, mean=mbes_meas_ranges, cov=self.meas_cov)
        else:
            print ("missing pings!")
            w_i = 0.
        return w_i


    def weight_avg(self, mbes_meas_ranges, mbes_sim_ranges ):
        if len(mbes_meas_ranges) == len(mbes_sim_ranges):
            w_i = 1./self.p_num
            #  for i in range(len(mbes_sim_ranges)):
            w_i *= math.exp(-(((mbes_sim_ranges - mbes_meas_ranges)**2).mean())/(2*self.meas_cov))
        else:
            print ("missing pings!")
            w_i = 0.
        return w_i

    def predict_meas(self):

        # Find particle's mbes pose without broadcasting/listening to tf transforms
        particle_tf = Transform()
        particle_tf.translation = self.p_pose.position
        particle_tf.rotation    = self.p_pose.orientation
        mat_part = matrix_from_tf(particle_tf)
        trans_mat = self.m2o_tf_mat.dot(mat_part.dot(self.mbes_tf_mat))

        trans = TransformStamped()
        trans.transform.translation.x = translation_from_matrix(trans_mat)[0]
        trans.transform.translation.y = translation_from_matrix(trans_mat)[1]
        trans.transform.translation.z = translation_from_matrix(trans_mat)[2]
        trans.transform.rotation = Quaternion(*quaternion_from_matrix(trans_mat))

        # Build MbesSimGoal to send to action server
        mbes_goal = MbesSimGoal()
        mbes_goal.mbes_pose.header.frame_id = self.map_frame
        mbes_goal.mbes_pose.header.stamp = rospy.Time.now()
        mbes_goal.mbes_pose.transform = trans.transform

        return mbes_goal


    def get_pose_vec(self):
        """
        Returns a list of particle pose elements
        [x, y, z, roll, pitch, yaw]

        :return: List of pose values
        :rtype: List
        """
        pose_vec = []
        pose_vec.append(self.p_pose.position.x)
        pose_vec.append(self.p_pose.position.y)
        pose_vec.append(self.p_pose.position.z)

        quat = (self.p_pose.orientation.x,
                self.p_pose.orientation.y,
                self.p_pose.orientation.z,
                self.p_pose.orientation.w)
        roll, pitch, yaw = euler_from_quaternion(quat)

        pose_vec.append(roll)
        pose_vec.append(pitch)
        pose_vec.append(yaw)

        return pose_vec

def pcloud2ranges(point_cloud, pose_s):
    ranges = []
    for p in pc2.read_points(point_cloud, field_names = ("x", "y", "z"), skip_nans=True):
            # starts at left hand side of particle's mbes
        dx = pose_s.position.x - p[0]
        dy = pose_s.position.y - p[1]
        dz = pose_s.position.z - p[2]
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