#!/usr/bin/env python3

import os

import rospy
import time
import numpy as np
from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from geometry_msgs.msg import Transform, Quaternion
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32, Header
from sensor_msgs.msg import PointCloud2
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

# import keyboard  # using module keyboard


class AUV_Recorder(object):

    def __init__(self):

        root_folder = '/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/slam/rbpf_slam/data/record_pf2train_gp/'

        # gt_odom_topic = 'odom' # Odometry
        # sim_odom_topic = '/sim_mbes' # Odometry

        particle_poses_topic = '/pf/particle_poses' # PoseArray
        average_pose_topic = '/pf/avg_pose' # PoseWithCovariance
        target_topic = '/target_top' # PointCloud2
        input_topic = '/input_top' # PointCloud2

        # Subscription to real mbes pings 
        # mbes_pings_top = '/gt/mbes_pings'
        # rospy.Subscriber(mbes_pings_top, PointCloud2, self.mbes_real_cb)

        # rospy.Subscriber(gt_odom_topic, Odometry, self.gt_odom_callback)
        # rospy.Subscriber(sim_odom_topic, PointCloud2, self.sim_odom_callback)
        rospy.Subscriber(target_topic, numpy_msg(Floats), self.target_cb)
        rospy.Subscriber(input_topic, numpy_msg(Floats), self.input_cb)
        rospy.Subscriber(particle_poses_topic, PoseArray, self.pose_array_callback)
        rospy.Subscriber(average_pose_topic, PoseWithCovarianceStamped, self.pose_w_cov_callback)

        dir_name = ('results_' + str(time.gmtime().tm_year) + '_' + str(time.gmtime().tm_mon) + '_' + str(time.gmtime().tm_mday) + '___'
                    + str(time.gmtime().tm_hour) + '_' + str(time.gmtime().tm_min) + '_' + str(time.gmtime().tm_sec) + '/')
        if root_folder[-1] != '/':
            dir_name = '/' + dir_name

        storage_path = root_folder + dir_name
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)

        # self.odom_file  = open(storage_path + 'true_pose.csv',      "w")
        self.pose_file  = open(storage_path + 'measured_pose.csv',  "w")
        self.array_file = open(storage_path + 'particle_poses.csv', "w")
        self.targets_file = storage_path + 'target_gp.npy'
        self.inputs_file = storage_path + 'inputs_gp.npy'
        # self.targets_file = open(storage_path + 'target_gp.csv', "w")
        # self.inputs_file = open(storage_path + 'inputs_gp.csv', "w")

        # self.odom_file.write ('time, pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w \n')
        self.pose_file.write ('time, pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w \n')
        self.array_file.write('time, pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w, idx \n')
        # self.targets_file.write('time, pos.x, pos.y, pos.z, quat.x, quat.y, quat.z, quat.w, idx \n')


    # def mbes_real_cb(self, msg):
    #     time_ = (str(msg.header.stamp.secs) + '.' + str(msg.header.stamp.nsecs))
    #     pose_ = (str(msg.pose.pose.position.x) + ', ' + str(msg.pose.pose.position.y)
    #                                                  + ', ' + str(msg.pose.pose.position.z))
    #     quat_ = (str(msg.pose.pose.orientation.x) + ', ' + str(msg.pose.pose.orientation.y)
    #                  + ', ' + str(msg.pose.pose.orientation.z) + ', ' + str(msg.pose.pose.orientation.w))

    #     self.odom_file.write(time_  + ', ' + pose_ + ', ' + quat_ + '\n')

    # def gt_odom_callback(self, msg):
    #     time_ = (str(msg.header.stamp.secs) + '.' + str(msg.header.stamp.nsecs))
    #     pose_ = (str(msg.pose.pose.position.x) + ', ' + str(msg.pose.pose.position.y)
    #                                                  + ', ' + str(msg.pose.pose.position.z))
    #     quat_ = (str(msg.pose.pose.orientation.x) + ', ' + str(msg.pose.pose.orientation.y)
    #                  + ', ' + str(msg.pose.pose.orientation.z) + ', ' + str(msg.pose.pose.orientation.w))

    #     self.odom_file.write(time_  + ', ' + pose_ + ', ' + quat_ + '\n')



    # def sim_odom_callback(self, msg):
    #     time_ = (str(msg.header.stamp.secs) + '.' + str(msg.header.stamp.nsecs))
    #     pose_ = (str(msg.pose.pose.position.x) + ', ' + str(msg.pose.pose.position.y)
    #                                                  + ', ' + str(msg.pose.pose.position.z))
    #     quat_ = (str(msg.pose.pose.orientation.x) + ', ' + str(msg.pose.pose.orientation.y)
    #                  + ', ' + str(msg.pose.pose.orientation.z) + ', ' + str(msg.pose.pose.orientation.w))

    #     self.odom_file.write(time_  + ', ' + pose_ + ', ' + quat_ + '\n')

    def target_cb(self, msg):
        # targets = msg.data
        np.save(self.targets_file, msg)
        # targets = (str(msg) + '   DONE')
        # self.targets_file.write(targets)

    def input_cb(self, msg):
        np.save(self.inputs_file, msg)
        # self.inputs = msg
        # self.inputs_file.write(str(msg))

    def pose_w_cov_callback(self, msg):
        time_ = (str(msg.header.stamp.secs) + '.' + str(msg.header.stamp.nsecs))
        pose_ = (str(msg.pose.pose.position.x) + ', ' + str(msg.pose.pose.position.y)
                                                     + ', ' + str(msg.pose.pose.position.z))
        quat_ = (str(msg.pose.pose.orientation.x) + ', ' + str(msg.pose.pose.orientation.y)
                     + ', ' + str(msg.pose.pose.orientation.z) + ', ' + str(msg.pose.pose.orientation.w))

        self.pose_file.write(time_  + ', ' + pose_ + ', ' + quat_ + '\n')



    def pose_array_callback(self, msg):
        # seq_  = ('seq: '  + str(msg.header.seq) + '\n')
        time_ = (str(msg.header.stamp.secs) + '.' + str(msg.header.stamp.nsecs))
        all_info = ''
        for idx, pose in enumerate(msg.poses):
            pose_ = (str(pose.position.x) + ', ' + str(pose.position.y)
                                        + ', ' + str(pose.position.z))
            quat_ = (str(pose.orientation.x) + ', ' + str(pose.orientation.y)
            + ', ' + str(pose.orientation.z) + ', ' + str(pose.orientation.w))
            all_info += (time_  + ', ' + pose_ + ', ' + quat_ + ', ' + str(idx) + '\n')

        self.array_file.write(all_info)


if __name__ == '__main__':

    rospy.init_node('record2train')
    try:
        AUV_Recorder()
        print("PF results recorder running")
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch recorder")

    rospy.spin()