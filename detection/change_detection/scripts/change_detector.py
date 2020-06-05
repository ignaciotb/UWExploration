#!/usr/bin/env python

import rospy

import numpy as np
import math
import matplotlib.pyplot as plt

import tf
import tf2_ros

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
import sensor_msgs.point_cloud2 as pc2

import message_filters

class ChangeDetector(object):

    def __init__(self):
        
        map_frame = rospy.get_param('~map_frame', 'map') # map frame_id
        odom_frame = rospy.get_param('~odom_frame', 'odom') 
        meas_model_as = rospy.get_param('~mbes_as', '/mbes_sim_server') # map frame_id
        auv_odom_top = rospy.get_param("~odometry_topic", '/odom')
        auv_mbes_top = rospy.get_param("~mbes_pings_topic", '/mbes')
        pf_pose_top = rospy.get_param("~average_pose_topic", '/avg_pose')
        pf_mbes_top = rospy.get_param("~average_mbes_topic", '/avg_mbes')

        self.auv_mbes = message_filters.Subscriber(auv_mbes_top, PointCloud2)
        self.auv_pose = message_filters.Subscriber(auv_odom_top, Odometry)  
        self.pf_mbes = message_filters.Subscriber(pf_mbes_top, PointCloud2)
        self.pf_pose = message_filters.Subscriber(pf_pose_top, PoseWithCovarianceStamped)  
        self.ts = message_filters.ApproximateTimeSynchronizer([self.auv_mbes, self.pf_mbes, 
                                                              self.auv_pose, self.pf_pose], 
                                                              10, slop=10.0,
                                                              allow_headerless=False)
        self.ts.registerCallback(self.detectionCB)
        
        plt.ion()
        plt.show()
        self.max_height = 30. # TODO: this should equal the n beams in ping
        self.new_msg = False
        first_msg = True
        self.waterfall =[] 
 
        while not rospy.is_shutdown():
            if self.new_msg:
                #  plt.imshow(waterfall, aspect='equal')
                plt.imshow(np.array(self.waterfall), norm=plt.Normalize(-30., 30.), aspect='equal')
                if first_msg:
                    first_msg = False
                    plt.colorbar()
                    plt.title("Bathymetry difference (m)")
                
                plt.pause(0.01)       

                self.new_msg = False
    
        #  rospy.spin()


    def pcloud2ranges(self, point_cloud, pose_s):
        ranges = []
        for p in pc2.read_points(point_cloud, field_names = ("x", "y", "z"), skip_nans=True):
                # starts at left hand side of particle's mbes
            dx = pose_s.position.x - p[0]
            dy = pose_s.position.y - p[1]
            dz = pose_s.position.z - p[2]
            dist = math.sqrt((dx**2 + dy**2 + dz**2))
            ranges.append(dist)
        return np.asarray(ranges)


    def detectionCB(self, auv_ping, pf_ping, auv_pose, pf_pose):
        try:
            auv_ping = self.pcloud2ranges(auv_ping, auv_pose.pose.pose)
            pf_ping = self.pcloud2ranges(pf_ping, pf_pose.pose.pose)
            #  print (auv_ping - pf_ping)

            self.waterfall.append((auv_ping - pf_ping))
            if len(self.waterfall)>self.max_height:
                self.waterfall.pop(0)

            self.new_msg = True
        
        except rospy.ROSInternalException:
            pass

if __name__ == '__main__':

    rospy.init_node('car_detector_node')
    try:
        ChangeDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not launch car detector node')
        pass
