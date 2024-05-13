#!/usr/bin/env python3

import rospy
import math
from ixblue_ins_msgs.msg import Ins
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
import numpy as np
from tf.transformations import quaternion_matrix, quaternion_from_matrix, quaternion_inverse
from tf.transformations import quaternion_multiply, quaternion_inverse, euler_from_quaternion, quaternion_from_euler
import tf
from geometry_msgs.msg import PointStamped, Quaternion, TransformStamped
from geodesy import utm
import tf2_ros
import message_filters
from sensor_msgs.msg import Imu

from scipy.spatial.transform import Rotation

class Ins2Dr():

    def __init__(self):

        ins_top = rospy.get_param("~lolo_ins", "/lolo/core/ins")
        imu_top = rospy.get_param("~lolo_imu", "/lolo/core/imu")
        odom_top = rospy.get_param("~lolo_odom", "/lolo/core/odom")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.utm_frame = rospy.get_param("~utm_frame", "utm")

        self.odom_pub = rospy.Publisher(odom_top, Odometry, queue_size=10)
        self.prev_heading = None
        self.now_heading = None
        self.t_prev = 0.
        self.t_now = 0.
        self.n = 0

        self.listener = tf.TransformListener()
        self.static_tf_bc = tf2_ros.StaticTransformBroadcaster()

        # rospy.Subscriber(ins_top, Ins, self.ins_cb, queue_size=100)

        self.ins_sub = message_filters.Subscriber(ins_top, Ins)
        self.imu_sub = message_filters.Subscriber(imu_top, Imu)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.ins_sub, self.imu_sub], 20, slop=20.0, allow_headerless=False
        )
        self.ts.registerCallback(self.ts_cb)


    def ts_cb(self, ins_msg, imu_msg):

        # Broadcast utm --> odom where vehicle 
        utm_lolo = utm.fromLatLong(ins_msg.latitude, ins_msg.longitude)
        
        if self.prev_heading == None:

            self.prev_heading = ins_msg.heading
            print("Init_heading: {}".format(self.prev_heading))
            
            rot = Rotation.from_euler('xyz', [0.,
                                              0.,
                                              90. - ins_msg.heading], degrees=True)
            
            transformStamped = TransformStamped()
            transformStamped.transform.translation.x = utm_lolo.easting
            transformStamped.transform.translation.y = utm_lolo.northing
            transformStamped.transform.translation.z = ins_msg.altitude
            transformStamped.transform.rotation = Quaternion(*rot.as_quat())
            transformStamped.header.frame_id = self.utm_frame
            transformStamped.child_frame_id = self.odom_frame
            transformStamped.header.stamp = rospy.Time.now()
            self.static_tf_bc.sendTransform(transformStamped)

            self.t_prev = ins_msg.header.stamp
            return
      
        # Check if the goal has been reached
        goal_point = PointStamped()
        goal_point.header.frame_id = self.utm_frame
        goal_point.header.stamp = rospy.Time(0)
        goal_point.point.x = utm_lolo.easting
        goal_point.point.y = utm_lolo.northing
        goal_point.point.z = 0.

        try:
            base_pose = self.listener.transformPoint(
                self.odom_frame, goal_point)
            
            odom_msg = Odometry()
            self.t_now = ins_msg.header.stamp
            odom_msg.header.frame_id = self.odom_frame
            odom_msg.child_frame_id = "lolo/base_link"
            odom_msg.header.stamp = self.t_now
            odom_msg.pose.pose.position.x = base_pose.point.x
            odom_msg.pose.pose.position.y = base_pose.point.y
            odom_msg.pose.pose.position.z = 0
            odom_msg.twist.twist.linear.x = ins_msg.speed_vessel_frame.x
            odom_msg.twist.twist.linear.y = ins_msg.speed_vessel_frame.y
            odom_msg.twist.twist.linear.z = 0. # Surface
            odom_msg.twist.twist.angular.z = imu_msg.angular_velocity.z
            
            self.odom_pub.publish(odom_msg)
            print("Sending odom")

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            print("Heading controller: Could not transform WP to base_link")
            pass

if __name__ == "__main__":
    
    rospy.init_node("ins_2_dr")

    ins_2_dr = Ins2Dr()
    while not rospy.is_shutdown():
        rospy.spin()
