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

from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from geometry_msgs.msg import Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, euler_from_quaternion

# For sim mbes action client
import actionlib
from auv_2_ros.msg import MbesSimGoal, MbesSimAction
from sensor_msgs.msg import PointCloud2

# Define
meas_freq = 5 # [Hz] to run mbes_sim
pcloud_pub_index = 2 # Which particle's mbes pointcloud to publish


class Particle():
    def __init__(self, index, map_frame, mbes_trans, broadcaster, static_broadcaster):
        self.index = index
        self.frame_id = "particle_" + str(index)
        self.mbes_frame_id = "particle_" + str(index) + "_mbes_link"

        self.weight = 1.
        self.pose = Pose()
        self.pose.orientation.w = 1.
        
        self.transform = TransformStamped() 
        self.transform.header.frame_id = map_frame
        self.transform.child_frame_id = self.frame_id
        self.transform.transform.rotation.w = 1 
        broadcaster.sendTransform(self.transform)

        self.mbes_trans = TransformStamped()
        self.mbes_trans.transform = mbes_trans.transform
        self.mbes_trans.header.frame_id = self.frame_id
        self.mbes_trans.child_frame_id = self.mbes_frame_id
        self.mbes_trans.header.stamp = rospy.Time.now()
        static_broadcaster.sendTransform(self.mbes_trans)


    def update(self, vel_vec, noise_vec, dt, broadcaster, true_pose):
        quat = (self.pose.orientation.x,
                self.pose.orientation.y,
                self.pose.orientation.z,
                self.pose.orientation.w)
        _, _, yaw = euler_from_quaternion(quat)

        self.pose.position.x += vel_vec[0] * dt * math.cos(yaw) + noise_vec[0] + vel_vec[1] * dt * math.sin(yaw)
        self.pose.position.y += vel_vec[0] * dt * math.sin(yaw) + noise_vec[1] + vel_vec[1] * dt * math.cos(yaw)
        yaw += vel_vec[2] * dt + noise_vec[2] # No need for remainder bc quaternion
        self.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, yaw))

        self.transform.transform.translation = self.pose.position
        self.transform.transform.rotation = self.pose.orientation
        self.transform.header.stamp = rospy.Time.now()
        broadcaster.sendTransform(self.transform)


        # Update particles weight
        # p_pose = [self.pose.position.x, self.pose.position.y, yaw]
        # -----------------------------
        # for j, tp in enumerate(true_pose):
        print(true_pose)
        # distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        distance = np.linalg.norm(p_pose - true_pose, axis=1)
        self.weight *= scipy.stats.norm(distance, sensor_std).pdf(observation[j])

        self.weight += 1.e-300 # avoid round-off to zero
        self.weight /= np.sum(self.weight) # normalize
        self.weight = np.log(self.weight)
    
# def update(particles, weights, observation, sensor_std, landmarks):
#     '''
#     Update particle weights
    
#     PARAMETERS
#      - particles:    Locations and headings of all particles
#      - weights:      Weights of all particles
#      - observation:  Observation of distances between robot and all landmarks
#      - sensor_std:   Standard deviation for error in sensor for observation
#      - landmarks:    Locations of all landmarks
    
#     DESCRIPTION
#     Set all weights to 1. For each landmark, calculate the distance between 
#     the particles and that landmark. Then, for a normal distribution with mean 
#     = distance and std = sensor_std, calculate the pdf for a measurement of observation. 
#     Multiply weight by pdf. If observation is close to distance, then the 
#     particle is similar to the true state of the model so the pdf is close 
#     to one so the weight stays near one. If observation is far from distance,
#     then the particle is not similar to the true state of the model so the 
#     pdf is close to zero so the weight becomes very small.   
    
#     The distance variable depends on the particles while the z parameter depends 
#     on the robot.
#     '''
#     weights.fill(1.)


class auv_pf():
    def __init__(self):
        # Read nnecessary ROS parameters
        param = rospy.search_param("map_frame")
        self.map_frame = rospy.get_param(param) # map frame_id
        param = rospy.search_param("odometry_topic")
        self.odom_top = rospy.get_param(param) # odometry msg topic (subscribed)
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
        self.predict_cov = list(map(float, cov_list)) # [xv_cov, yv_cov, yaw_v_cov]

        # Initialize class/callback variables
        self.pred_odom = None
        self.time = None
        self.old_time = None
        self.pos_ = PoseArray()
        self.pos_.header.frame_id = self.map_frame        

        # Initialize particle poses publisher
        self.pf_pub = rospy.Publisher(self.pose_array_top, PoseArray, queue_size=10)
        # Initialize sim_mbes pointcloud publisher
        self.pcloud_pub = rospy.Publisher('/devel_sim_mbes_pcloud', PointCloud2, queue_size=10)

        # Initialize tf listener and broadcaster
        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        rospy.sleep(1) # Must keep! Or static particle_mbes transforms don't broadcast
                
        try:
            # Confirm mbes_link & base_link are in the correct order!!!
            rospy.loginfo("Waiting for transform from base_link to mbes_link")
            self.mbes_trans = self.tfBuffer.lookup_transform('hugin/mbes_link', 'hugin/base_link', rospy.Time.now(), rospy.Duration(10))
            rospy.loginfo("Transform locked from base_link to mbes_link")
        except:
            rospy.loginfo("ERROR: Could not lookup transform from base_link to mbes_link")
        
        # Initialize list of particles
        self.particles = []
        for i in range(self.pc):
            self.particles.append(Particle(i+1, self.map_frame, self.mbes_trans, self.broadcaster, self.static_broadcaster))

        # Initialize connection to MbesSim action server
        self.ac_mbes = actionlib.SimpleActionClient('/mbes_sim_server',MbesSimAction)
        rospy.loginfo("Waiting for MbesSim action server")
        self.ac_mbes.wait_for_server()
        rospy.loginfo("Connected MbesSim action server")

        # Establish subscription to odometry message | Last because this will start callback running
        rospy.Subscriber(self.odom_top, Odometry, self.odom_callback)
        rospy.sleep(0.5) # CAN ADD DURATION INSTEAD? 
        

    def odom_callback(self,msg):
        self.pred_odom = msg
        self.time = self.pred_odom.header.stamp.secs + self.pred_odom.header.stamp.nsecs*10**-9 
        if self.old_time and self.time > self.old_time:
            self.predict()
            self.pub_()
        self.old_time = self.time



    def predict(self):
        # Adding gaussian noice
        pred_noice =  self.process_noise()
        # Unpack odometry message
        dt = self.time - self.old_time
        xv = self.pred_odom.twist.twist.linear.x
        yv = self.pred_odom.twist.twist.linear.y
        # vel = np.sqrt(np.power(xv,2) + np.power(yv,2))
        yaw_v = self.pred_odom.twist.twist.angular.z
        vel_vec = [xv, yv, yaw_v]
        # True pose to use for the weights
        quat = (self.pred_odom.pose.pose.orientation.x,
                self.pred_odom.pose.pose.orientation.y,
                self.pred_odom.pose.pose.orientation.z,
                self.pred_odom.pose.pose.orientation.w)
        _, _, yaw = euler_from_quaternion(quat)
        true_pose = [self.pred_odom.pose.pose.position.x, self.pred_odom.pose.pose.position.y, yaw]
        
        # Update particles pose estimate
        for i in range(len(self.particles)):
            particle = self.particles[i]
            particle.update(vel_vec, pred_noice[i,:], dt, self.broadcaster, true_pose)


    def pub_(self):
        self.pos_.poses = []
        for pt in self.particles:
            self.pos_.poses.append(pt.pose)
        # Publish particles with time odometry was received
        self.pos_.header.stamp.secs = int(self.time)
        self.pos_.header.stamp.nsecs = (self.time - int(self.time))*10**9
        self.pf_pub.publish(self.pos_)


    def process_noise(self):
        cov_ = np.zeros((3,3))
        cov_[0,0] = self.predict_cov[0]
        cov_[1,1] = self.predict_cov[1]
        cov_[2,2] = self.predict_cov[2]
        var = np.diagonal(cov_)
        return np.sqrt(var)*np.random.randn(self.pc, 3)


    def measurement(self):
        # Right now this only runs for the first particle
        # Can be expanded to all particles once it for sure works
        # particle = self.particles[0]
        for particle in self.particles:
            mbes_pcloud = self.pf2mbes(particle)
            # Only publish one particle's mbes for debugging/visualization purposes
            if particle.index == pcloud_pub_index:
                self.pcloud_pub.publish(mbes_pcloud)


    def pf2mbes(self, particle_):

        trans = self.tfBuffer.lookup_transform(self.map_frame, particle_.mbes_frame_id , rospy.Time())

        # Build MbesSimGoal to send to action server
        mbes_goal = MbesSimGoal()
        mbes_goal.mbes_pose.header.frame_id = self.map_frame
        mbes_goal.mbes_pose.child_frame_id = particle_.mbes_frame_id # The particles will be in a child frame to the map
        mbes_goal.mbes_pose.header.stamp = rospy.Time.now()
        mbes_goal.mbes_pose.transform = trans.transform

        # Get result from action server
        self.ac_mbes.send_goal(mbes_goal)
        rospy.loginfo("Waiting for MbesSim action Result")
        self.ac_mbes.wait_for_result()
        rospy.loginfo("Got MbesSim action Result")
        mbes_res = self.ac_mbes.get_result()

        # Pack result into PointCloud2
        mbes_pcloud = PointCloud2()
        mbes_pcloud = mbes_res.sim_mbes
        mbes_pcloud.header.frame_id = self.map_frame
        return mbes_pcloud


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
    
    # rospy.spin()


if __name__ == '__main__':
    main()

