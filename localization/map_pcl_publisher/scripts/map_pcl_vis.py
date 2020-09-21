#!/usr/bin/env python

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

class MapPCLPublisher(object):

    def __init__(self):
        
        self.cloud_path = rospy.get_param('~map_cloud_path')
        self.pings_path = rospy.get_param('~pings_path')
        self.map_frame = rospy.get_param('~map_frame')
        self.map_pub = rospy.Publisher('/map', PointCloud2, queue_size=1)

        cloud = xyz_data.cloud.parse_file(self.cloud_path)
	pings = std_data.mbes_ping.read_data(self.pings_path)

	R = rot.from_euler("zyx", [pings[0].heading_, pings[0].pitch_, 0.]).as_dcm()
        pos = pings[0].pos_
        R_inv = R.transpose()
        pos_inv = R_inv.dot(pos)
        cloud[:] = [R_inv.dot(p) - pos_inv for p in cloud]

        mbes_pcloud = PointCloud2()
        header = Header()
        header.frame_id = self.map_frame
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1)]
        
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            header.stamp = rospy.Time.now()
            mbes_pcloud = point_cloud2.create_cloud(header, fields, cloud)
            self.map_pub.publish(mbes_pcloud)
            
            rate.sleep()

if __name__ == '__main__':

    rospy.init_node('map_pcl_publisher')
    try:
        MapPCLPublisher()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not launch map pcl publisher')
        pass


