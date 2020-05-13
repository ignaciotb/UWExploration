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
# from functools import partial
# nprocs = mp.cpu_count()
# pool = mp.Pool(processes=nprocs)
# print(sys.version)

# Define (add to launch file at some point)
meas_freq = 1 # [Hz] to run mbes_sim
std = 0.1 # Noise in the multibeam (tunable parameter)
use_log_weights = True # Boolean
use_N_eff_from_paper = False # Boolean
"""
Test using new vs old N_eff calc and decide which is better
"""

class auv_pf():
    def __init__(self):
        # Read nnecessary ROS parameters
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
        for i in range(self.pc):
            self.particles.append(Particle(i+1, mbes_matrix, process_cov=predict_cov, map_frame=self.map_frame)) # particle index starts from 1
        """
        Attempt at using multiprocessing
        Either need to alter functions to use
        Python 2.7 multiporcessing or upgrade env to Python3
        """
        # self.particles = pool.map(partial(Particle, map_frame=self.map_frame), [i+1 for i in range(self.pc)])

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
        # vel = np.sqrt(np.power(xv,2) + np.power(yv,2))
        yaw_v = self.pred_odom.twist.twist.angular.z
        vel_vec = [xv, yv, yaw_v]

        # Update particles pose estimate
        """
        FOR MULTIPROCESSING:

        Could append dt to vel_vec and calc noise in particle.pred_update
        OR also append pred_noice[idx,:] to vel_vec and have it be a pass in vector
        """
        for idx, particle in enumerate(self.particles):
            particle.pred_update(vel_vec, dt)


    def measurement(self):
        mbes_meas_ranges = self.pcloud2ranges(self.mbes_true_pc, self.pred_odom.pose.pose)
        """
        Should pred_odom pose not be ussed b/c we won't actually know it ???
        Maybe we should take average of partricles here?
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
        for particle in self.particles:
            mbes_pcloud = particle.simulate_mbes(self.ac_mbes)
            mbes_sim_ranges = self.pcloud2ranges(mbes_pcloud, particle.pose)
            self.pcloud_pub.publish(mbes_pcloud)

            try: # Sometimes there is no result for mbes_sim_ranges
                mse = ((mbes_meas_ranges - mbes_sim_ranges)**2).mean()
                """
                Calculate regular weight AND log weight for now
                """
                weight = math.exp(-mse/(2*std**2))
                log_w = C - mse/(2*std**2)
            except:
                rospy.loginfo('Caught exception in auv_pf.measurement() function')
                log_w = -1.e100 # A very large negative value
                weight = 1.e-300 # avoid round-off to zero

            weights.append(weight)
            log_weights.append(log_w)

        if use_log_weights:
            norm_factor = logsumexp(log_weights)
            weights_ = np.asarray(log_weights)
            weights_ -= norm_factor
            weights_ = np.exp(weights_)
        else:
            weights_ = np.asarray(weights)
        
        self.resample(weights_)
        self.average_pf_pose()


    def resample(self, weights):

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

        if use_N_eff_from_paper:
            N_eff = 1/np.sum(np.square(weights)) # From paper
        else:
            N_eff = self.pc - len(lost) # old version

        if N_eff < self.pc/2: # Threshold to perform resampling
            for i in range(len(lost)): # Perform resampling
                self.particles[lost[i]].pose = deepcopy(self.particles[dupes[i]].pose)
                """
                Consider adding noise to resampled particle
                """
        else:
            rospy.loginfo('Number of effective particles too high - not resampling')


    def average_pf_pose(self):
        x_, y_, z_ = [], [], []
        roll_, pitch_, yaw_ = [], [], []

        """
        FOR MULTIPROCESSING:

        Each particle could return a vector of [x,y,z,roll,pitch,yaw]
        And then we turn all the vectors into one  numpy array and do
        our row averaging operations afterwards
        """
        for particle in self.particles:
            x_.append(particle.pose.position.x)
            y_.append(particle.pose.position.y)
            z_.append(particle.pose.position.z)

            quat = (particle.pose.orientation.x,
                    particle.pose.orientation.y,
                    particle.pose.orientation.z,
                    particle.pose.orientation.w)
            roll, pitch, yaw = euler_from_quaternion(quat)
            roll_.append(roll)
            pitch_.append(pitch)
            yaw_.append(yaw)
            
        pf_pose = PoseWithCovarianceStamped()
        pf_pose.header.frame_id = self.map_frame

        pf_pose.pose.pose.position.x = sum(x_) / len(x_)
        pf_pose.pose.pose.position.y = sum(y_) / len(y_)
        pf_pose.pose.pose.position.z = sum(z_) / len(z_)

        roll  = sum(roll_) / len(roll_)
        pitch = sum(pitch_) / len(pitch_)
        """
        Average of list of angles (e.g. yaw) creates
        issues when heading towards pi because pi and
        negative pi are next to eachother, but average
        out to zero (opposite direction of heading)
        """
        abs_yaw_ = map(abs, yaw_)
        if min(abs_yaw_) > math.pi/2:
            pos_yaw_ = [x + 2*math.pi if x<0 else x for x in yaw_]
            yaw = sum(pos_yaw_) / len(pos_yaw_)
        else:
            yaw = sum(yaw_) / len(yaw_)

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

    meas_rate = rospy.Rate(meas_freq) # Hz
    while not rospy.is_shutdown():
        pf.measurement()
        meas_rate.sleep()


if __name__ == '__main__':
    main()