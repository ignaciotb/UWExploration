#!/usr/bin/env python3

import numpy as np

import rospy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2

import open3d as o3d

class MapPCLPublisher(object):

    def __init__(self):

        self.cloud_path = rospy.get_param('~map_cloud_path')
        self.sift_cloud_path = rospy.get_param('~map_sift_path')
        self.gp_cloud_path = rospy.get_param('~map_gp_path')
        self.map_frame = rospy.get_param('~map_frame')
        self.map_pub = rospy.Publisher('/map_mbes', PointCloud2, queue_size=1, latch=True)
        self.map_sift_pub = rospy.Publisher(
            '/map_sift', PointCloud2, queue_size=1, latch=True)
        self.map_gp_pub = rospy.Publisher(
            '/map_gp', PointCloud2, queue_size=1, latch=True)
        raw_data = False

        print("Map from MBES pings")
        # cloud = np.load(self.cloud_path)
        pcd = o3d.io.read_point_cloud(self.cloud_path)
        pcd = pcd.uniform_down_sample(every_k_points=3)
        cloud = np.asarray(pcd.points)
        
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
            
            # gp_pcloud = PointCloud2()
            header = Header()
            header.frame_id = self.map_frame
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1)]
            
            gp_pcloud = point_cloud2.create_cloud(header, fields, gp_cloud)
            gp_cloud = None

        if self.sift_cloud_path != "":    
            pcd = o3d.io.read_point_cloud(self.sift_cloud_path)
            sift_cloud = np.asarray(pcd.points)  

            print("Map from SIFT features ", sift_cloud.shape)
            # sift_pcloud = PointCloud2()
            header = Header()
            header.frame_id = self.map_frame
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1)]
            
            sift_pcloud = point_cloud2.create_cloud(header, fields, sift_cloud)
            gp_cloud = None


        rate = rospy.Rate(0.5)
        # while not rospy.is_shutdown():
        header.stamp = rospy.Time.now()
        self.map_pub.publish(mbes_pcloud)

        if self.gp_cloud_path:
            self.map_gp_pub.publish(gp_pcloud)
        
        if self.sift_cloud_path:
            self.map_sift_pub.publish(sift_pcloud)

        rospy.spin()

            # rate.sleep()

if __name__ == '__main__':

    rospy.init_node('map_pcl_publisher')
    try:
        MapPCLPublisher()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not launch map pcl publisher')
        pass
