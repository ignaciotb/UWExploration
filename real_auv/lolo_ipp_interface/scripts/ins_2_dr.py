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
        if self.prev_heading == None:

            self.prev_heading = ins_msg.heading
            print("Init_heading: {}".format(self.prev_heading))
            
            utm_odom = utm.fromLatLong(ins_msg.latitude, ins_msg.longitude)
            rot = Rotation.from_euler('xyz', [0.,
                                              0.,
                                              90. - ins_msg.heading], degrees=True)
            
            transformStamped = TransformStamped()
            transformStamped.transform.translation.x = utm_odom.easting
            transformStamped.transform.translation.y = utm_odom.northing
            transformStamped.transform.translation.z = ins_msg.altitude
            transformStamped.transform.rotation = Quaternion(*rot.as_quat())
            transformStamped.header.frame_id = self.utm_frame
            transformStamped.child_frame_id = self.odom_frame
            transformStamped.header.stamp = rospy.Time.now()
            self.static_tf_bc.sendTransform(transformStamped)

            self.t_prev = ins_msg.header.stamp
            return

        # TODO Nacho: parse lat lon and orientation to odom for GT comparison
        odom_msg = Odometry()
        self.t_now = ins_msg.header.stamp
        odom_msg.header.frame_id = self.map_frame
        odom_msg.header.stamp = self.t_now
        odom_msg.twist.twist.linear.x = ins_msg.speed_vessel_frame.x
        odom_msg.twist.twist.linear.y = ins_msg.speed_vessel_frame.y
        odom_msg.twist.twist.linear.z = 0. # Surface

        # Compute angular velocities from orientations
        # dt = (self.t_now - self.t_prev).to_sec()

        # self.now_heading = ins_msg.heading
        # d_heading = self.now_heading - self.prev_heading
        # v_ang_heading = (d_heading * 2 * math.pi / 360.) / dt
        # self.t_prev = self.t_now
        # self.prev_heading = self.now_heading
        # print('heading vel ', v_ang_heading)

        # odom_msg.twist.twist.angular.z = -v_ang_heading
        odom_msg.twist.twist.angular.z = imu_msg.angular_velocity.z
        
        # # The heading in our framework is the DR yaw relative to the map
        # heading_t = ins_msg.heading - self.init_heading
        # rot = Rotation.from_euler('xyz', [ins_msg.roll,
        #                                   ins_msg.pitch,
        #                                   heading_t], degrees=True)

        # odom_msg.pose.pose.orientation.x = rot.as_quat()[0]
        # odom_msg.pose.pose.orientation.y = rot.as_quat()[1]
        # odom_msg.pose.pose.orientation.z = rot.as_quat()[2]
        # odom_msg.pose.pose.orientation.w = rot.as_quat()[3]

        # Downsample publication
        # self.n += 1
        #if self.n % 5 == 0:
        self.odom_pub.publish(odom_msg)
        #    self.n = 0

if __name__ == "__main__":
    
    rospy.init_node("ins_2_dr")

    ins_2_dr = Ins2Dr()
    while not rospy.is_shutdown():
        rospy.spin()
