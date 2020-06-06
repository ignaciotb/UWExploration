#!/usr/bin/env python

import rospy

import cv2
import scipy.misc
from PIL import Image

import numpy as np
import math
import matplotlib.pyplot as plt

import tf
import tf2_ros
from tf.transformations import rotation_matrix, rotation_from_matrix, translation_matrix, translation_from_matrix, quaternion_matrix, quaternion_from_matrix

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Transform, PoseWithCovarianceStamped
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
        self.ts.registerCallback(self.pingCB)

        # Initialize tf listener
        tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tfBuffer)
        try:
            rospy.loginfo("Waiting for transforms")
            mbes_tf = tfBuffer.lookup_transform('hugin/base_link', 'hugin/mbes_link',
                                                rospy.Time(0), rospy.Duration(20.))
            self.base2mbes_mat = self.matrix_from_tf(mbes_tf)
            
            m2o_tf = tfBuffer.lookup_transform(map_frame, odom_frame,
                                               rospy.Time(0), rospy.Duration(20.))
            self.m2o_mat = self.matrix_from_tf(m2o_tf)

            rospy.loginfo("Transforms locked - Car detector node")
        except:
            rospy.loginfo("ERROR: Could not lookup transform from base_link to mbes_link")

 
        plt.ion()
        plt.show()
        self.max_height = 30. # TODO: this should equal the n beams in ping
        self.new_msg = False
        first_msg = True
        self.waterfall =[] 
 
        while not rospy.is_shutdown():
            if self.new_msg:
                plt.imshow(np.array(self.waterfall), norm=plt.Normalize(0., 60.),
                           cmap='gray', aspect='equal')
                if first_msg:
                    first_msg = False
                    plt.colorbar()
                    plt.title("Bathymetry difference (m)")

                plt.pause(0.01)
                
                #  if len(self.waterfall)==self.max_height:
                    #  img = self.car_detection(np.array(self.waterfall))
            #
                    #  print "img"
                    #  cv2.imshow("hola", np.float32(img))
                    #  cv2.waitKey(1)
            self.new_msg = False
    
        #  rospy.spin()

    def car_detection(self, img_array):
        #  im = cv2.imread("blob.jpg", cv2.IMREAD_GRAYSCALE)
        
        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()
         
        # Change thresholds
        params.minThreshold = 100;
        params.maxThreshold = 5000;
         
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 200
         
        # Filter by Circularity
        params.filterByCircularity = False
        params.minCircularity = 0.785
         
        # Filter by Convexity
        params.filterByConvexity = False
        params.minConvexity = 0.87
         
        # Filter by Inertia
        #params.filterByInertia = True
        #params.minInertiaRatio = 0.01

        # Set up the detector with default parameters.
        detector = cv2.SimpleBlobDetector(params)
        
        # Detect blobs.
        img_array = np.float32(img_array)
        #  img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        rgb = Image.fromarray(img_array)
        #  keypoints = detector.detect(rgb)
        
        #  plt.imshow(rgb, norm=plt.Normalize(0., 60.),
                            #  cmap='gray', aspect='equal')
                #
        plt.pause(0.01)
        return rgb
        
#
        #  # Draw detected blobs as red circles.
        #  # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        #  im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#
        #  # Show keypoints
        #  cv2.imshow("Keypoints", im_with_keypoints)
        #  rospy.sleep(0.1)
        #  #  cv2.waitKey(0)

    def pcloud2ranges(self, point_cloud, tf_mat):
        angle, direc, point = rotation_from_matrix(tf_mat)
        R = rotation_matrix(angle, direc, point)
        rot_inv = R[np.ix_([0,1,2],[0,1,2])].transpose()
        
        t = translation_from_matrix(tf_mat)
        t_inv = rot_inv.dot(t)

        ranges = []
        for p in pc2.read_points(point_cloud, field_names = ("x", "y", "z"), skip_nans=True):
            p_part = rot_inv.dot(p) - t_inv
            ranges.append(np.linalg.norm(p_part))
        
        return np.asarray(ranges)


    def pingCB(self, auv_ping, pf_ping, auv_pose, pf_pose):
        try:
            particle_tf = Transform()
            particle_tf.translation = auv_pose.pose.pose.position
            particle_tf.rotation    = auv_pose.pose.pose.orientation
            tf_mat = self.matrix_from_tf(particle_tf)
            m2auv = np.matmul(self.m2o_mat, np.matmul(tf_mat, self.base2mbes_mat))
            auv_ping_ranges = self.pcloud2ranges(auv_ping, m2auv)

            #  auv_ping = self.pcloud2ranges(auv_ping, auv_pose.pose.pose)
            #  pf_ping = self.pcloud2ranges(pf_ping, pf_pose.pose.pose)
            #  print (auv_ping_ranges)

            self.waterfall.append(abs(auv_ping_ranges - auv_ping_ranges))
            if len(self.waterfall)>self.max_height:
                self.waterfall.pop(0)

            self.new_msg = True
        
        except rospy.ROSInternalException:
            pass
    
    def matrix_from_tf(self, transform):
        """
        Converts a geometry_msgs/Transform or
        geometry_msgs/TransformStamped into a 4x4
        transformation matrix

        :param transform: Transform from parent->child frame
        :type transform: geometry_msgs/Transform(Stamped)
        :return: Transform as 4x4 matrix
        :rtype: Numpy array (4x4)
        """
        if transform._type == 'geometry_msgs/TransformStamped':
            transform = transform.transform

        trans = (transform.translation.x,
                 transform.translation.y,
                 transform.translation.z)
        quat_ = (transform.rotation.x,
                 transform.rotation.y,
                 transform.rotation.z,
                 transform.rotation.w)

        tmat = translation_matrix(trans)
        qmat = quaternion_matrix(quat_)

        return np.dot(tmat, qmat)
if __name__ == '__main__':

    rospy.init_node('car_detector_node')
    try:
        ChangeDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not launch car detector node')
        pass
