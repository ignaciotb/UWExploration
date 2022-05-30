#!/usr/bin/env python3
"""
Node to transform the LaserScan message from LoLo's MBES
to a Pointcloud2 in the ENU frame.
"""
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped
import tf2_ros
import tf
import numpy as np
import ros_numpy
import rospy

__author__ = "Aldo Teran"
__author_email = "aldot@kth.se"
__license__ = "MIT"
__status__ = "Development"

#TODO: Namespaces?
class ScanToPtcloud:
    """
    Class to turn the MBES laserscan message to a poitncloud2.
    """
    def __init__(self):
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()

        # Wait for buffer then get rotation from mbes to ENU frame
        #TODO: This should be done in a modular manner
        try:
            trans = self.tf_buffer.lookup_transform('lolo/base_link', 'lolo/mbes_link',
                                                    rospy.Time())
            rospy.loginfo("Found TF base_link -> mbes_link")
            quat = tf.transformations.quaternion_from_euler(0.0,-0.036128,0.0)
            trans.transform.rotation.x = quat[0]
            trans.transform.rotation.y = quat[1]
            trans.transform.rotation.z = quat[2]
            trans.transform.rotation.w = quat[3]
            self.mbes_rot = trans
        except:
            rospy.logwarn("Could not find TF base_link -> mbes_link, setting default.")
            trans = TransformStamped()
            quat = tf.transformations.quaternion_from_euler(0.0,-0.036128,0.0)
            trans.transform.rotation.x = quat[0]
            trans.transform.rotation.y = quat[1]
            trans.transform.rotation.z = quat[2]
            trans.transform.rotation.w = quat[3]
            trans.transform.translation.x = 2.523
            trans.transform.translation.y = 0.0155
            trans.transform.translation.z = -0.302
            self.mbes_rot = trans
        self.mbes_rot.header.stamp = rospy.Time.now()
        self.mbes_rot.header.frame_id = "lolo/base_link"
        self.mbes_rot.child_frame_id = "lolo/enu/mbes_link"
        self.tf_broadcaster.sendTransform(self.mbes_rot)
        rospy.loginfo("Publishing TF lolo/base_link -> lolo/enu/mbes_link.")

        self.pointcloud_pub = rospy.Publisher("lolo/mbes/enu/bathy_cloud",
                                              PointCloud2, queue_size=0)

        rospy.Subscriber("lolo/mbes/bathy_points", LaserScan, self.ls_callback_)

    def ls_callback_(self, data):
        """
        Callback for the LaserScan message. Beam ranges and bearings
        are defined as follows: around the X axis (swatch), min angle starts at
        Y and max angle ends at -Y axis; range (depth) is measured from the origin
        along the X axis.
        """
        # Build array with beam angles
        angles = np.arange(data.angle_max, data.angle_min,
                           -data.angle_increment)
        # Get the ranges
        ranges = np.asarray(data.ranges)
        # normalize intensities
        intensities = np.asarray(data.intensities)
        intensities = np.uint8(255 * (intensities / np.max(intensities)))

        # Build the recarray with points and intensities
        pointcloud = np.recarray((1,len(ranges)),dtype=[('x', np.float32),
                                                        ('y', np.float32),
                                                        ('z', np.float32),
                                                        ('intensity', np.uint8)])
        pointcloud['x'] = np.zeros(len(ranges))
        pointcloud['y'] = ranges * np.sin(angles)
        pointcloud['z'] = -ranges * np.cos(angles)
        pointcloud['intensity'] = intensities
        # Do some magic
        pointcloud_msg = ros_numpy.point_cloud2.array_to_pointcloud2(pointcloud)
        pointcloud_msg.header.stamp = data.header.stamp
        pointcloud_msg.header.frame_id = "lolo/enu/mbes_link"

        # Publish
        self.pointcloud_pub.publish(pointcloud_msg)

def main():
    rospy.init_node("mbes_scan_2_pointcloud")
    rospy.loginfo("Starting mbes scan to pointcloud2 node.")
    scan_2_ptcloud = ScanToPtcloud()
    while not rospy.is_shutdown():
        rospy.spin()

if __name__ == "__main__":
    main()













