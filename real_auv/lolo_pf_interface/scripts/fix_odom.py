#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
import numpy as np
from tf.transformations import quaternion_matrix, quaternion_from_matrix
from tf.transformations import quaternion_multiply, quaternion_inverse, euler_from_quaternion

class FixOdom:

    def __init__(self):

        odom_top_in = rospy.get_param("~lolo_odom_in", "/lolo/dr/odom")
        odom_top_out = rospy.get_param("~lolo_odom_out", "/lolo/dr/odom_fixed")
        rospy.Subscriber(odom_top_in, Odometry, self.odom_cb, queue_size=100)

        self.odom_pub = rospy.Publisher(odom_top_out, Odometry, queue_size=0)
        self.init_odom_count = 0
        self.old_time = rospy.Time.now().to_sec()
        self.time = rospy.Time.now().to_sec()

    def odom_cb(self, odom_t):
        odom_t.header.frame_id = "lolo/odom"
        odom_t.child_frame_id = "lolo/base_link"

        self.time = odom_t.header.stamp.to_sec()

        if self.init_odom_count == 0:
            self.prev_odom = odom_t
            self.init_odom_count += 1

        if self.old_time and self.time > self.old_time:

            dt = self.time - self.old_time

            q_array = np.array([odom_t.pose.pose.orientation.x, odom_t.pose.pose.orientation.y,
                                odom_t.pose.pose.orientation.z, odom_t.pose.pose.orientation.w])
            prev_q_array = np.array([self.prev_odom.pose.pose.orientation.x, self.prev_odom.pose.pose.orientation.y,
                                    self.prev_odom.pose.pose.orientation.z, self.prev_odom.pose.pose.orientation.w])

            vel_t_x = (odom_t.pose.pose.position.x -
                        self.prev_odom.pose.pose.position.x) / dt
            vel_t_y = (odom_t.pose.pose.position.y -
                        self.prev_odom.pose.pose.position.y) / dt
            vel_t_z = (odom_t.pose.pose.position.z -
                        self.prev_odom.pose.pose.position.z) / dt
            vel_t = np.array([vel_t_x, vel_t_y, vel_t_z])
            prev_q_mat = quaternion_matrix(prev_q_array)
            prev_q_mat = prev_q_mat[0:3, 0:3]
            vel_rel = np.linalg.inv(prev_q_mat) @ np.reshape(vel_t, (3, 1))

            odom_t.twist.twist.linear.x = vel_rel[0, 0]
            odom_t.twist.twist.linear.y = vel_rel[1, 0]
            odom_t.twist.twist.linear.z = vel_rel[2, 0]

            delta_q = quaternion_multiply(q_array, quaternion_inverse(prev_q_array))
            roll_step, pitch_step, yaw_step = euler_from_quaternion(delta_q)
            odom_t.twist.twist.angular.x = roll_step / dt
            odom_t.twist.twist.angular.y = pitch_step / dt
            odom_t.twist.twist.angular.z = yaw_step / dt
            
            self.prev_odom = odom_t

        self.old_time = self.time

        self.odom_pub.publish(odom_t)



if __name__ == "__main__":
    rospy.init_node("fix_odom_node")

    fix_odom = FixOdom()
    while not rospy.is_shutdown():
        rospy.spin()
