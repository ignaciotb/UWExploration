#!/usr/bin/env python3

import numpy as np
import tf2_ros
from auvlib.bathy_maps import mesh_map
import os

import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2
import open3d as o3d

class MapPCLPublisher(object):

    def __init__(self):

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.cloud_path = rospy.get_param('~map_cloud_path')
        self.map_frame = rospy.get_param('~map_frame')
        self.map_pub = rospy.Publisher('/map_mbes', PointCloud2, queue_size=1)

        # Wait for buffer then get rotation from utm to map/odom frame
        try:
            trans = self.tf_buffer.lookup_transform('utm', 'map',
                                                    rospy.Time(),
                                                    rospy.Duration(10))
            rospy.loginfo("Map Publisher: Found TF utm -> map for map generation")
            self.utm_offset = {'x': trans.transform.translation.x,
                               'y': trans.transform.translation.y}
        except:
            rospy.logerr("Map Publisher: Could not find TF utm -> map, setting to zero.")
            self.utm_offset = {'x': 0.0, 'y': 0.0}

        # Process the map and save it as an npy for visualization and npz mesh
        # for raytraycing.
        rospy.loginfo("Map Publisher: Processing map %s", self.cloud_path)
        pcd = o3d.io.read_point_cloud(self.cloud_path)
        pcd = np.asarray(pcd.points)
        pcd[:,0] -= self.utm_offset['x']
        pcd[:,1] -= self.utm_offset['y']
        # rospy.loginfo("Map Publisher: Saving npy map as KBerg_mission_map.npy")
        # np.save(os.path.join("/home/aldoteran/slam_ws/maps/",
                             # "KBerg_mission_map.npy"), pcd)
        V, F, bounds = mesh_map.mesh_from_dtm_cloud(pcd, 0.6)
        rospy.loginfo("Map Publisher: Saving map mesh as KBerg_mission_map.npz")
        np.savez(os.path.join("/home/aldoteran/slam_ws/maps/",
                              "KBerg_mission_map.npz"), V=V, F=F, bounds=bounds)

        pointcloud = np.recarray((1,len(pcd)),dtype=[('x', np.float32),
                                                     ('y', np.float32),
                                                     ('z', np.float32)])
        pointcloud['x'] = pcd[:,0]
        pointcloud['y'] = pcd[:,1]
        pointcloud['z'] = pcd[:,2]

        mbes_pcloud = ros_numpy.point_cloud2.array_to_pointcloud2(pointcloud)
        mbes_pcloud.header.frame_id = self.map_frame

        # Release memory
        pcd = None

        #TODO: maybe better to make it a latched topic instead of publishing often
        rate = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            mbes_pcloud.header.stamp = rospy.Time.now()
            self.map_pub.publish(mbes_pcloud)
            rospy.loginfo("Map Publisher: Publishing map %s on frame %s", self.cloud_path,
                          self.map_frame)
            rate.sleep()

if __name__ == '__main__':

    rospy.init_node('map_pcl_publisher')
    try:
        MapPCLPublisher()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not launch map pcl publisher')
        pass
