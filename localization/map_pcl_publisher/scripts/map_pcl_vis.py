#!/usr/bin/env python3

import numpy as np
from auvlib.data_tools import gsf_data, std_data, csv_data, xyz_data
from auvlib.bathy_maps import mesh_map, base_draper
import configargparse
import math
import os
import sys

from scipy import stats
from scipy.spatial.transform import Rotation as rot

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
import open3d as o3d

class MapPCLPublisher(object):

    def __init__(self):

        self.cloud_path = rospy.get_param('~map_cloud_path')
        self.gp_cloud_path = rospy.get_param('~map_gp_path')
        self.map_frame = rospy.get_param('~map_frame')
        self.map_pub = rospy.Publisher('/map_mbes', PointCloud2, queue_size=1)
        self.map_gp_pub = rospy.Publisher('/map_gp', PointCloud2, queue_size=1)
        raw_data = False

        print("Map from MBES pings")
        cloud = np.load(self.cloud_path)

        mbes_pcloud = PointCloud2()
        header = Header()
        header.frame_id = self.map_frame
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]

        mbes_pcloud = point_cloud2.create_cloud(header, fields, cloud)
        cloud = None

        if self.gp_cloud_path != "":
            print("Map from GP")
            gp_cloud = np.load(self.gp_cloud_path)
            gp_cloud = gp_cloud[:,0:3]

            gp_pcloud = PointCloud2()
            header = Header()
            header.frame_id = self.map_frame
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1)]

            gp_pcloud = point_cloud2.create_cloud(header, fields, gp_cloud)
            gp_cloud = None

        #TODO: maybe better to make it a latched topic instead of publishing often
        rate = rospy.Rate(0.01)
        while not rospy.is_shutdown():
            header.stamp = rospy.Time.now()
            self.map_pub.publish(mbes_pcloud)
            rospy.loginfo("Publishing map %s on frame %s", self.cloud_path,
                          self.map_frame)

            if self.gp_cloud_path:
                self.map_gp_pub.publish(gp_pcloud)

            rate.sleep()

if __name__ == '__main__':

    rospy.init_node('map_pcl_publisher')
    try:
        MapPCLPublisher()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not launch map pcl publisher')
        pass
