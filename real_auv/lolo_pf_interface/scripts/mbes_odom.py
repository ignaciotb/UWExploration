#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool, Header
from nav_msgs.msg import Odometry
import numpy as np
from tf.transformations import quaternion_matrix, quaternion_from_matrix, quaternion_from_euler
from tf.transformations import quaternion_multiply, quaternion_inverse, euler_from_quaternion
from scipy.spatial.transform import Rotation as rot
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Transform, Quaternion
import tf

class FixOdom:

    def __init__(self):

        odom_top_in = rospy.get_param("~lolo_odom_in", "/lolo/dr/odom")
        odom_top_out = rospy.get_param("~lolo_odom_out", "/lolo/dr/odom_fixed")
        self.mbes_frame = rospy.get_param("~mbes_link", "mbes_link")
        self.odom_frame = rospy.get_param("~odom_frame", "odom")
        self.base_frame = rospy.get_param("~base_frame", "base_link")
        rospy.Subscriber(odom_top_in, Odometry, self.odom_cb, queue_size=100)

        self.odom_pub = rospy.Publisher(odom_top_out, Odometry, queue_size=0)
        self.init_odom_count = 0
        self.old_time = Header.stamp
        self.time = Header.stamp

        # finished_top = rospy.get_param("~survey_finished_top", '/survey_finished')
        # self.synch_pub = rospy.Subscriber(finished_top, Bool, self.save_cb)
        self.storage_path = rospy.get_param("~results_path")
        self.heading_noise = rospy.get_param("~heading_noise", '/odom_corrupted')
        
        # Corrupted DR for experiments
        self.cr_odom_top = rospy.get_param("~corrupted_odom_topic", 0.0)
        self.corr_dr_pub = rospy.Publisher(self.cr_odom_top, PoseStamped, queue_size=0)
        self.corrupted_pose_t = [0.]*6

        self.listener = tf.TransformListener()
        rospy.Timer(rospy.Duration(0.01), self.timer_callback)

        self.tf_time = 0.
        self.odom_cnt = 0
        self.odom_cnt_prev = 0
        self.odom_msg = Odometry()
        # self.tf_time_prev = 0.

    def odom_cb(self, odom_msg):
        self.time = odom_msg.header.stamp
        self.odom_msg = odom_msg
        self.odom_cnt += 1

    def timer_callback(self, event):

        try:
            (trans_mbes, rot_mbes) = self.listener.lookupTransform(
                self.odom_frame, self.mbes_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        self.tf_time = self.listener.getLatestCommonTime(self.odom_frame, self.mbes_frame)
            
        odom_t = Odometry()
        # print(trans)
        odom_t.header.frame_id = self.odom_frame
        odom_t.child_frame_id = self.base_frame
        odom_t.header.stamp = self.time
        odom_t.pose.pose.position.x = trans_mbes[0]
        odom_t.pose.pose.position.y = trans_mbes[1]
        odom_t.pose.pose.position.z = trans_mbes[2]

        odom_t.pose.pose.orientation.x = rot_mbes[0]
        odom_t.pose.pose.orientation.y = rot_mbes[1]
        odom_t.pose.pose.orientation.z = rot_mbes[2]
        odom_t.pose.pose.orientation.w = rot_mbes[3]

        # Original velocities seem unusable
        odom_t.twist.twist.linear.x = self.odom_msg.twist.twist.linear.x
        odom_t.twist.twist.linear.y = self.odom_msg.twist.twist.linear.y
        odom_t.twist.twist.linear.z = self.odom_msg.twist.twist.linear.z
        odom_t.twist.twist.angular.x = self.odom_msg.twist.twist.angular.x
        odom_t.twist.twist.angular.y = self.odom_msg.twist.twist.angular.y
        odom_t.twist.twist.angular.z = self.odom_msg.twist.twist.angular.z

        if self.init_odom_count == 0:
            self.prev_odom = odom_t
            self.init_odom_count += 1

            self.corrupted_pose_t[0] = odom_t.pose.pose.position.x
            self.corrupted_pose_t[1] = odom_t.pose.pose.position.y
            self.corrupted_pose_t[2] = odom_t.pose.pose.position.z

            q_t = np.array([odom_t.pose.pose.orientation.x, odom_t.pose.pose.orientation.y,
                            odom_t.pose.pose.orientation.z, odom_t.pose.pose.orientation.w])
            roll_t, pitch_t, yaw_t = euler_from_quaternion(q_t)
            self.corrupted_pose_t[3] = roll_t
            self.corrupted_pose_t[4] = pitch_t
            self.corrupted_pose_t[5] = yaw_t

            self.old_time = self.time
        
        if self.init_odom_count != 0 and self.time.to_sec() > self.old_time.to_sec():

            dt = self.time.to_sec() - self.old_time.to_sec()
            self.odom_cnt_prev = self.odom_cnt
            # dt = 0.05

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
            vel_rel = np.matmul(np.linalg.inv(prev_q_mat), vel_t)

            odom_t.twist.twist.linear.x = vel_rel[0]
            odom_t.twist.twist.linear.y = vel_rel[1]
            odom_t.twist.twist.linear.z = vel_rel[2]

            delta_q = quaternion_multiply(q_array, quaternion_inverse(prev_q_array))
            roll_step, pitch_step, yaw_step = euler_from_quaternion(delta_q)
            odom_t.twist.twist.angular.x = roll_step / dt
            odom_t.twist.twist.angular.y = pitch_step / dt

            # # Corrupt heading velocity      
            odom_t.twist.twist.angular.z = yaw_step / dt
            odom_t.twist.twist.angular.z += np.sqrt(self.heading_noise) * np.random.randn()

            self.prev_odom = odom_t
            self.odom_pub.publish(odom_t)

            ############# Compute instance of corrupted DR here
            vel_rot = np.array([odom_t.twist.twist.angular.x,
                                odom_t.twist.twist.angular.y,
                                odom_t.twist.twist.angular.z])
            rot_t = np.array(self.corrupted_pose_t[3:6]) + vel_rot * dt

            # rot_t[1] = ((rot_t[1]) + 2 * np.pi) % (2 * np.pi)
            # rot_t[2] = ((rot_t[2]) + 2 * np.pi) % (2 * np.pi)
            rot_t[0] = 0.
            rot_t[1] = 0.
            rot_t[2] = ((rot_t[2]) + 2 * np.pi) % (2 * np.pi)
            self.corrupted_pose_t[3:6] = rot_t

            # Linear motion
            vel_p = np.array([odom_t.twist.twist.linear.x,
                            odom_t.twist.twist.linear.y,
                            odom_t.twist.twist.linear.z])
            
            rot_mat_t = rot.from_euler("xyz", rot_t).as_matrix()
            step_t = np.matmul(rot_mat_t, vel_p * dt)

            self.corrupted_pose_t[0] += step_t[0]
            self.corrupted_pose_t[1] += step_t[1]
            # Seems to be a problem when integrating depth from Ping vessel, so we just read it
            self.corrupted_pose_t[2] = odom_t.pose.pose.position.z

            # Publish
            self.corrupted_pose = PoseStamped()
            self.corrupted_pose.pose.position.x = self.corrupted_pose_t[0]
            self.corrupted_pose.pose.position.y = self.corrupted_pose_t[1]
            self.corrupted_pose.pose.position.z = self.corrupted_pose_t[2]            
            self.corrupted_pose.pose.orientation = Quaternion(*quaternion_from_euler(rot_t[0],
                                                                                    rot_t[1],
                                                                                    rot_t[2]))
            # self.corrupted_pose.pose.orientation = Quaternion(*q_array)
            self.corrupted_pose.header.frame_id = odom_t.header.frame_id
            self.corrupted_pose.header.stamp = odom_t.header.stamp
            self.corr_dr_pub.publish(self.corrupted_pose)
        
        self.old_time = self.time

        # self.tf_time_prev = self.tf_time       


if __name__ == "__main__":
    rospy.init_node("fix_odom_node")

    fix_odom = FixOdom()
    while not rospy.is_shutdown():
        rospy.spin()
