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

# import action 
from auv_2_ros.action import MbesSim
import actionlib

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from sensor_msgs.msg import PointCloud2 # For the action

class Particle():
    def __init__(self):
        self.pose = Pose()
        self.pose.orientation.w = 1.
        self.weight = 1.

    def update(self, vel_vec, noise_vec, dt):
        quat = (self.pose.orientation.x,
                self.pose.orientation.y,
                self.pose.orientation.z,
                self.pose.orientation.w)
        _, _, yaw = euler_from_quaternion(quat)

        self.pose.position.x += vel_vec[0] * dt * math.cos(yaw) + noise_vec[0]
        self.pose.position.y += vel_vec[0] * dt * math.sin(yaw) + noise_vec[1]
        yaw += vel_vec[1] * dt + noise_vec[2] # No need for remainder bc quaternion
        self.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, yaw))

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

        # Initialize list of particles
        self.particles = []
        for _ in range(self.pc):
            self.particles.append(Particle())

        # Initialize particle poses publisher
        self.pf_pub = rospy.Publisher(self.pose_array_top, PoseArray, queue_size=10)

        # Establish subscription to odometry message
        rospy.Subscriber(self.odom_top, Odometry, self.odom_callback)
        rospy.sleep(0.5) # CAN ADD DURATION INSTEAD?      
    
    def odom_callback(self,msg):
        self.pred_odom = msg
        self.time = self.pred_odom.header.stamp.secs + self.pred_odom.header.stamp.nsecs*10**-9 
        if self.old_time and self.time > self.old_time:
            self.predict()
            self.pf2mbes() # sim MBES for each particle
            self.pub_()
        self.old_time = self.time

    def predict(self):
        # Adding gaussian noice
        pred_noice =  self.process_noise()
        # Unpack odometry message
        dt = self.time - self.old_time
        xv = self.pred_odom.twist.twist.linear.x
        yv = self.pred_odom.twist.twist.linear.y
        vel = np.sqrt(np.power(xv,2) + np.power(yv,2))
        yaw_v = self.pred_odom.twist.twist.angular.z
        vel_vec = [vel, yaw_v]
        
        # Update particles pose estimate
        for i in range(len(self.particles)):
            particle = self.particles[i]
            particle.update(vel_vec, pred_noice[i,:], dt)

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

    # def cb(req):
    #     # I want to send that each partcle pose is a Mbes pose, 
    #     # so, from action. mbe_pose is particle pose and sim_mbes is what will be recieved on the other hand
    #     s = rospy.Service('pf_mbes',MbesSim, cb )
    #     rospy.spin()


    def pf2mbes(self):
        ac = actionlib.SimpleActionClient('pf_mbes',MbesSim)
        trans = geometry_msgs.Transform()
        tf_mbes_map = tf.Transformer
        mbes_goal = auv_2_ros.MbesSimGoal()
        mbes_goal.mbes_pose.header.frame_id = self.map_frame
        mbes_goal.mbes_pose.header.stamp = int(self.time)
        # mbes_goal.mbes_pose.header.stamp.nsecs = (self.time - int(self.time))*10**9
        mbes_goal.mbes_pose.transform.translation = self.pos_.position
        mbes_goal.mbes_pose.transform.rotation = self.pos_.orientation
        # mbes_goal.mbes_pose.transform = trans
        ac.send_goal(mbes_goal)


def main():
    # Initialize ROS node
    rospy.init_node('auv_pf', anonymous=True)
    rospy.loginfo("Successful initilization of node")

    # Create particle filter class
    auv_pf()
    rospy.loginfo("Particle filter class successfully created")

    rospy.spin()


if __name__ == '__main__':
    main()
