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
from scipy.special import logsumexp # For log weights
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

# Import Particle() class
from auv_particle import Particle, matrix_from_tf

# Multiprocessing
# import time # For evaluating mp improvements
# import multiprocessing as mp
# from functools import partial # Might be useful with mp
# from pathos.multiprocessing import ProcessingPool as Pool

# Define (add to launch file at some point)
T_meas = 2 # [s] Period between MBES scans
std = 0.1 # Noise in the multibeam (tunable parameter)
use_log_weights = True # Boolean
use_N_eff_from_paper = False # Boolean
"""
Test using new vs old N_eff calc and decide which is better
"""

class auv_pf():
    def __init__(self):
        # Read necessary ROS parameters
        param = rospy.search_param("map_frame")
        self.map_frame = rospy.get_param(param) # map frame_id
        param = rospy.search_param("odometry_topic")
        self.odom_top = rospy.get_param(param) # odometry msg topic (subscribed)
        param = rospy.search_param("mbes_pings_topic")
        self.mbes_pings_top = rospy.get_param(param) # mbes_pings msg topic (subscribed)
        param = rospy.search_param("particle_poses_topic")
        self.pose_array_top = rospy.get_param(param) # Particle pose array topic (published)
        param = rospy.search_param("particle_count")
        self.pc = rospy.get_param(param) # Particle Count
        
        # Read motion covariance values (and convert to float list) 
        param = rospy.search_param("motion_covariance")
        cov_string = rospy.get_param(param)
        cov_string = cov_string.replace('[','')
        cov_string = cov_string.replace(']','')
        cov_list = list(cov_string.split(", "))
        predict_cov = list(map(float, cov_list)) # [xv_cov, yv_cov, yaw_v_cov]

        # Initialize class/callback variables
        self.pred_odom = None
        self.time = None
        self.old_time = None
        self.pos_ = PoseArray()
        self.pos_.header.frame_id = self.map_frame        
        self.mbes_true_pc = None

        # Initialize particle poses publisher
        self.pf_pub = rospy.Publisher(self.pose_array_top, PoseArray, queue_size=10)
        # Initialize average of poses publisher
        self.avg_pub = rospy.Publisher('/avg_pf_pose', PoseWithCovarianceStamped, queue_size=10)
        # Initialize sim_mbes pointcloud publisher
        self.pcloud_pub = rospy.Publisher('/particle_mbes_pclouds', PointCloud2, queue_size=10)

        # Initialize tf listener (and broadcaster)
        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)
        # self.broadcaster = tf2_ros.TransformBroadcaster()
                
        try:
            rospy.loginfo("Waiting for transform from base_link to mbes_link")
            mbes_tf = self.tfBuffer.lookup_transform('hugin/mbes_link', 'hugin/base_link', rospy.Time.now(), rospy.Duration(10))
            mbes_matrix = matrix_from_tf(mbes_tf)
            rospy.loginfo("Transform locked from base_link to mbes_link - pf node")
        except:
            rospy.loginfo("ERROR: Could not lookup transform from base_link to mbes_link")
        
        # Initialize list of particles
        self.particles = []
        for i in range(self.pc): # particle index starts from 1
            self.particles.append(Particle(i+1, mbes_matrix, process_cov=predict_cov, map_frame=self.map_frame))

        # Initialize connection to MbesSim action server
        self.ac_mbes = actionlib.SimpleActionClient('/mbes_sim_server',MbesSimAction)
        rospy.loginfo("Waiting for MbesSim action server")
        self.ac_mbes.wait_for_server()
        rospy.loginfo("Connected MbesSim action server")

        # Establish subscription to mbes pings message
        rospy.Subscriber(self.mbes_pings_top, PointCloud2, self._mbes_callback)
        # Establish subscription to odometry message | Last because this will start callback & prediction running
        rospy.Subscriber(self.odom_top, Odometry, self.odom_callback)
        rospy.sleep(0.5) # CAN ADD DURATION INSTEAD? 


    def _mbes_callback(self, msg):
        self.mbes_true_pc = msg

    def odom_callback(self,msg):
        self.pred_odom = msg
        self.time = self.pred_odom.header.stamp.secs + self.pred_odom.header.stamp.nsecs*10**-9 
        if self.old_time and self.time > self.old_time:
            self.predict()
            self._posearray_pub()
        self.old_time = self.time


    def predict(self):
        # Unpack odometry message
        dt = self.time - self.old_time
        xv = self.pred_odom.twist.twist.linear.x
        yv = self.pred_odom.twist.twist.linear.y
        yaw_v = self.pred_odom.twist.twist.angular.z

        z = self.pred_odom.pose.pose.position.z
        qx = self.pred_odom.pose.pose.orientation.x
        qy = self.pred_odom.pose.pose.orientation.y
        qz = self.pred_odom.pose.pose.orientation.z
        qw = self.pred_odom.pose.pose.orientation.w
        update_vec = [dt, xv, yv, yaw_v, z, qx, qy, qz, qw]

        """
        Failed attempts at using multiprocessing for pred_update
        """
        # nprocs = mp.cpu_count()
        # # pool = mp.Pool(processes=nprocs)
        # pool = Pool(nprocs)
        # pool.map(_pred_update_helper, ((particle, update_vec) for particle in self.particles))

        # # multi_result = [pool.apply_async(_pred_update_helper, (particle, update_vec)) for particle in self.particles]
        # # result = [p.get() for p in multi_result]

        pose_list = []
        for particle in self.particles:
            particle.pred_update(update_vec)
            """
            particle.get_pose_vec() function could be combined
            with particle.pred_update() if that seems cleaner
            """
            pose_vec = particle.get_pose_vec()
            pose_list.append(pose_vec)

        self.average_pose(pose_list) # Calculate the average


    def measurement(self):
        mbes_meas_ranges = self.pcloud2ranges(self.mbes_true_pc, self.pred_odom.pose.pose)

        log_weights = []
        weights = []

        for particle in self.particles:
            mbes_pcloud = particle.simulate_mbes(self.ac_mbes)
            mbes_sim_ranges = self.pcloud2ranges(mbes_pcloud, particle.pose)
            self.pcloud_pub.publish(mbes_pcloud)

            w, log_w = particle.weight(mbes_meas_ranges, mbes_sim_ranges) # calculating particles weights
            weights.append(w)
            log_weights.append(log_w)

        if use_log_weights:
            norm_factor = logsumexp(log_weights)
            weights_ = np.asarray(log_weights)
            weights_ -= norm_factor
            weights_ = np.exp(weights_)
        else:
            weights_ = np.asarray(weights)

        for particle in self.particles: # Overrighting, can do self.particles[0].resample instead
            N_eff, lost, dupes = particle.resample(weights, self.pc) # For resampling

        if N_eff < self.pc/2: # Threshold to perform resampling
            for i in range(len(lost)): # Perform resampling
                # Faster to do separately than using deepcopy()
                self.particles[lost[i]].pose.position.x = self.particles[dupes[i]].pose.position.x
                self.particles[lost[i]].pose.position.y = self.particles[dupes[i]].pose.position.y
                self.particles[lost[i]].pose.position.z = self.particles[dupes[i]].pose.position.z
                self.particles[lost[i]].pose.orientation.x = self.particles[dupes[i]].pose.orientation.x
                self.particles[lost[i]].pose.orientation.y = self.particles[dupes[i]].pose.orientation.y
                self.particles[lost[i]].pose.orientation.z = self.particles[dupes[i]].pose.orientation.z
                self.particles[lost[i]].pose.orientation.w = self.particles[dupes[i]].pose.orientation.w
                """
                Consider adding noise to resampled particle
                """
        else:
            rospy.loginfo('Number of effective particles too high - not resampling')

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
        self.avg_pub.publish(pf_pose)


    def pcloud2ranges(self, point_cloud, pose):
        ranges = []
        for p in pc2.read_points(point_cloud, field_names = ("x", "y", "z"), skip_nans=True):
            # starts at left hand side of particle's mbes
            dx = pose.position.x - p[0]
            dy = pose.position.y - p[1]
            dz = pose.position.z - p[2]
            dist = math.sqrt((dx**2 + dy**2 + dz**2))
            ranges.append(dist)
        return np.asarray(ranges)


    def _posearray_pub(self):
        self.pos_.poses = []
        for particle in self.particles:
            self.pos_.poses.append(particle.pose)
        # Publish particles with time odometry was received
        self.pos_.header.stamp.secs = int(self.time)
        self.pos_.header.stamp.nsecs = (self.time - int(self.time))*10**9
        self.pf_pub.publish(self.pos_)


def main():
    # Initialize ROS node
    rospy.init_node('auv_pf', anonymous=True)
    rospy.loginfo("Successful initilization of node")

    # Create particle filter class
    pf = auv_pf()
    rospy.loginfo("Particle filter class successfully created")

    meas_rate = rospy.Rate(1/float(T_meas))
    while not rospy.is_shutdown():
        pf.measurement()
        meas_rate.sleep()


# def _pred_update_helper(args):
#     """
#     Worker function for multiprocessing
#     prediction update
#     """
#     particle, update_vec = args
#     particle.pred_update(update_vec)


if __name__ == '__main__':
    main()