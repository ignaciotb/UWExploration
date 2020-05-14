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
from geometry_msgs.msg import Quaternion, Transform, TransformStamped
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf.transformations import translation_matrix, translation_from_matrix
from tf.transformations import quaternion_matrix, quaternion_from_matrix

# For sim mbes action client
import actionlib
from auv_2_ros.msg import MbesSimGoal, MbesSimAction
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

# Define (add to launch file at some point)
T_meas = 2 # [s] Period between MBES scans
std = 0.1 # Noise in the multibeam (tunable parameter)
use_log_weights = True # Boolean
use_N_eff_from_paper = False # Boolean


class Particle():
    def __init__(self, index, mbes_tf_matrix, process_cov=[0., 0., 0.], map_frame='map'):
        self.index = index # index starts from 1
        # self.weight = 1.
        self.pose = Pose()
        self.pose.orientation.w = 1.
        self.map_frame = map_frame
        self.mbes_tf_mat = mbes_tf_matrix
        self.process_cov = np.asarray(process_cov)


    def pred_update(self, update_vec):
        """
        Unpack update_vec
        update_vec = [dt, xv, yv, yaw_v, z, qx, qy, qz, qw]
        """
        dt = update_vec[0]
        vel_vec = update_vec[1:4]
        depth = update_vec[4]
        true_quat = tuple(update_vec[5:])
        """
        I think there should be a faster
        way to compute noise_vec
        """
        noise_vec = (np.sqrt(self.process_cov)*np.random.randn(1, 3)).flatten()

        particl_quat = (self.pose.orientation.x,
                        self.pose.orientation.y,
                        self.pose.orientation.z,
                        self.pose.orientation.w)
        _, _, yaw = euler_from_quaternion(particl_quat)

        self.pose.position.x += vel_vec[0] * dt * math.cos(yaw) + noise_vec[0] + vel_vec[1] * dt * math.sin(yaw)
        self.pose.position.y += vel_vec[0] * dt * math.sin(yaw) + noise_vec[1] + vel_vec[1] * dt * math.cos(yaw)
        yaw += vel_vec[2] * dt + noise_vec[2]
        """
        depth, roll, & pitch are known from sensors
        """
        self.pose.position.z = depth
        roll, pitch, _ = euler_from_quaternion(true_quat)

        self.pose.orientation = Quaternion(*quaternion_from_euler(roll, pitch, yaw))

    def weight(self, mbes_meas_ranges, sim_ranges, pc ):
        """
        Should pred_odom pose not be used b/c we won't actually know it?
        Maybe this isn't relevant once we have better weight functions
        """
        log_weights = []
        weights = []
        """
        If trying to use log weights instead of regular weights:
            log(w) = C - (1/(2*std**2))*sum(z_hat - z)**2
            C = #son * log(sqrt(2*pi*std**2))
        """
        C = len(mbes_meas_ranges)*math.log(math.sqrt(2*math.pi*std**2))
        """
        FOR MULTIPROCESSING:

        mbes_ac is now the only pass in variable if we make a vector of
        mse values and then calc weights using numpy array functions
        after multiproc is done
        """
        
        for p in range(pc):
            mbes_sim_ranges = sim_ranges[p]
            try: # Sometimes there is no result for mbes_sim_ranges
                mse = ((mbes_meas_ranges - mbes_sim_ranges)**2).mean()
                """
                Calculate regular weight AND log weight for now
                """
                w = math.exp(-mse/(2*std**2))
                log_w = C - mse/(2*std**2)
            except:
                rospy.loginfo('Caught exception in auv_pf.measurement() function')
                log_w = -1.e100 # A very large negative value
                w = 1.e-300 # avoid round-off to zero

            weights.append(w)
            log_weights.append(log_w)

        if use_log_weights:
            norm_factor = logsumexp(log_weights)
            weights_ = np.asarray(log_weights)
            weights_ -= norm_factor
            weights_ = np.exp(weights_)
        else:
            weights_ = np.asarray(weights)
        
        return weights_

    def resample(self, weights, particle):

        # Define cumulative density function
        cdf = np.cumsum(weights)
        cdf /= cdf[cdf.size-1]
        # Multinomial resampling
        r = np.random.rand(self.pc,1)
        indices = []
        for i in range(self.pc):
            indices.append(np.argmax(cdf >= r[i]))
        indices.sort()

        keep = list(set(indices)) # set of particles resampled (independent of count)
        lost = [i for i in range(self.pc) if i not in keep] # particle poses to forget
        dupes = indices[:] # particle poses to replace the forgotten
        for i in keep:
            dupes.remove(i)
        """
        Choose only one of these N_eff calcs to retain
        """
        # if use_N_eff_from_paper:
        N_eff = 1/np.sum(np.square(weights)) # From paper
        # else:
        #     N_eff = self.pc - len(lost) # old version

        if N_eff < self.pc/2: # Threshold to perform resampling
            for i in range(len(lost)): # Perform resampling
                # Faster to do separately than using deepcopy()
                particle[lost[i]].pose.position.x = particle[dupes[i]].pose.position.x
                particle[lost[i]].pose.position.y = particle[dupes[i]].pose.position.y
                particle[lost[i]].pose.position.z = particle[dupes[i]].pose.position.z
                particle[lost[i]].pose.orientation.x = particle[dupes[i]].pose.orientation.x
                particle[lost[i]].pose.orientation.y = particle[dupes[i]].pose.orientation.y
                particle[lost[i]].pose.orientation.z = particle[dupes[i]].pose.orientation.z
                particle[lost[i]].pose.orientation.w = particle[dupes[i]].pose.orientation.w
                """
                Consider adding noise to resampled particle
                """
        else:
            rospy.loginfo('Number of effective particles too high - not resampling')

        return particle


    def average_pose(self, pose_list):
        """
        Get average pose of particles and
        publish it as PoseWithCovarianceStamped

        :param pose_list: List of lists containing pose
                        of all particles in form
                        [x, y, z, roll, pitch, yaw]
        :type pose_list: list
        """
        pf_pose = PoseWithCovarianceStamped()
        pf_pose.header.frame_id = self.map_frame

        poses_array = np.array(pose_list)
        ave_pose = poses_array.mean(axis = 0)

        pf_pose.pose.pose.position.x = ave_pose[0]
        pf_pose.pose.pose.position.y = ave_pose[1]
        """
        If z, roll, and pitch can stay as read directly from
        the odometry message there is no need to average them.
        We could just read from any arbitrary particle
        """
        pf_pose.pose.pose.position.z = ave_pose[2]
        roll  = ave_pose[3]
        pitch = ave_pose[4]
        """
        Average of yaw angles creates
        issues when heading towards pi because pi and
        negative pi are next to eachother, but average
        out to zero (opposite direction of heading)
        """
        yaws = poses_array[:,5]
        if np.abs(yaws).min() > math.pi/2:
            yaws[yaws < 0] += 2*math.pi
        yaw = yaws.mean()

        pf_pose.pose.pose.orientation = Quaternion(*quaternion_from_euler(roll, pitch, yaw))
        pf_pose.header.stamp = rospy.Time.now()
        return pf_pose


    def simulate_mbes(self, mbes_ac):

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
        mbes_ac.send_goal(mbes_goal)
        mbes_ac.wait_for_result()
        mbes_res = mbes_ac.get_result()

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