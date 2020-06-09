#!/usr/bin/env python

import rospy

import cv2
import scipy.misc
from PIL import Image

import numpy as np
import scipy.interpolate
import math
import matplotlib.pyplot as plt

import tf
import tf2_ros
from tf.transformations import rotation_matrix, rotation_from_matrix, translation_matrix, translation_from_matrix, quaternion_matrix, quaternion_from_matrix

from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, PoseArray, Transform, PoseWithCovarianceStamped, Vector3
import sensor_msgs.point_cloud2 as pc2

import message_filters

class ChangeDetector(object):

    def __init__(self):

        map_frame = rospy.get_param('~map_frame', 'map') # map frame_id
        odom_frame = rospy.get_param('~odom_frame', 'odom')
        meas_model_as = rospy.get_param('~mbes_as', '/mbes_sim_server') # map frame_id
        auv_odom_top = rospy.get_param("~odometry_topic", '/odom')
        auv_mbes_top = rospy.get_param("~mbes_pings_topic", '/mbes')
        auv_exp_mbes_top = rospy.get_param("~expected_mbes_topic", '/expected_mbes')
        pf_pose_top = rospy.get_param("~average_pose_topic", '/avg_pose')
        detection_top = rospy.get_param("~detection_topic", '/detection_pose')

        self.auv_mbes = message_filters.Subscriber(auv_mbes_top, PointCloud2)
        self.exp_mbes = message_filters.Subscriber(auv_exp_mbes_top, PointCloud2)
        self.auv_pose = message_filters.Subscriber(auv_odom_top, Odometry)
        self.pf_pose = message_filters.Subscriber(pf_pose_top, PoseWithCovarianceStamped)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.auv_mbes, self.exp_mbes,
                                                              self.auv_pose, self.pf_pose],
                                                              10, slop=3.0,
                                                              allow_headerless=False)

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


        # Register cb after tf is locked
        self.ts.registerCallback(self.pingCB)

        plt.ion()
        plt.show()

        self.ping_cnt = 0
        self.scale = 4
        self.max_height = 250 # TODO: this should equal the n beams in ping
        self.new_msg = False
        first_msg = True
        self.waterfall =[]
        self.active_auv_poses = []
        self.active_pf_pings = []

        #Init detection publisher
        self.detection_pb = rospy.Publisher(detection_top, PoseArray, queue_size=10)

        rospy.loginfo("Change detection node created")

        detections = PoseArray()
        detections.header.frame_id = 'map'
        while not rospy.is_shutdown():
            if self.new_msg:
                #Visualize
                if len(self.waterfall)==self.max_height:
                    waterfall_detect, centroids_row, centroids_col = self.car_detection(np.array(self.waterfall), self.scale)
                    # Visualize detection markers
                    if len(centroids_row) > 0:
                        for i in range(len(centroids_row)):
                            row = centroids_row[i]
                            col = centroids_col[i]
                            
                            det_msg = Pose()
                            #  det_msg.pose = self.active_auv_poses[row].pose
                            det_msg.position.x = self.active_pf_pings[row][col][0]
                            det_msg.position.y = self.active_pf_pings[row][col][1]
                            det_msg.position.z = self.active_pf_pings[row][col][2]
                            det_msg.orientation.y = 0.7071068
                            det_msg.orientation.w = 0.7071068
                            detections.poses.append(det_msg)
                    
                    waterfall_img = waterfall_detect

                else:
                    waterfall_img = self.waterfall    
                    
                plt.imshow(np.array(self.waterfall), norm=plt.Normalize(0., 5.),
                        cmap='gray', aspect='equal', origin = "lower")
                    
                if first_msg:
                    first_msg = False
                    #  plt.colorbar()
                    plt.title("Bathymetry difference (m)")

                plt.pause(0.01)

            self.new_msg = False
            self.detection_pb.publish(detections)

    def car_detection(self, img_array, scale):
        # Turn numpy array into cv2 image (and make bigger)
        img_array = np.float32(img_array)
        f = scipy.interpolate.RectBivariateSpline(np.linspace(0 ,1, np.size(img_array, 0)),
                                                  np.linspace(0, 1, np.size(img_array, 1)), img_array)
        img_array = f(np.linspace(0, 1, scale*np.size(img_array, 0)), 
                      np.linspace(0, 1, scale*np.size(img_array, 1)))
        gray_img = cv2.normalize(src=img_array, dst=None, alpha=255, beta=0,
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # Setup SimpleBlobDetector parameters.
        params = cv2.SimpleBlobDetector_Params()

        params.minThreshold = 1
        params.maxThreshold = 5000

        params.filterByArea = True 
        params.minArea = 20

        params.filterByCircularity = False
        params.minCircularity = 0.785

        params.filterByConvexity = False
        params.minConvexity = 0.87
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs.
        keypoints = detector.detect(gray_img)
        rows, cols = [], []
        if len(keypoints) != 0:
            for keypoint in keypoints:
                rows.append(int(keypoint.pt[1])/scale)
                cols.append(int(keypoint.pt[0])/scale)
        im_with_keypoints = cv2.drawKeypoints(gray_img, keypoints, np.array([]), (0,0,255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #  cv2.imshow('image', im_with_keypoints)
        #  cv2.waitKey(0)

        # Turn cv2 image back to numpy array, scale it down, and return
        out_img_array = np.empty((np.size(img_array,0), np.size(img_array,1) ,3), dtype=float)
        for i in range(np.size(im_with_keypoints,2)):
            f = scipy.interpolate.RectBivariateSpline(np.linspace(0 ,255, np.size(im_with_keypoints, 0)),
                                    np.linspace(0,255, np.size(im_with_keypoints, 1)), im_with_keypoints[:,:,i])
            out_img_array[:,:,i] = f(np.linspace(0, 255, np.size(img_array, 0)),
                                     np.linspace(0, 255, np.size(img_array, 1)))
        
        out_img_array = out_img_array.astype(np.float)
        return out_img_array, rows, cols

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

    def ping2ranges(self, point_cloud):
        ranges = []
        for p in pc2.read_points(point_cloud, field_names = ("x", "y", "z"), skip_nans=True):
            ranges.append(np.linalg.norm(p))

        return np.asarray(ranges)

    def ping2vecs(self, point_cloud, tf_mat):
        
        ranges = []
        for p in pc2.read_points(point_cloud, field_names = ("x", "y", "z"), skip_nans=True):
            ranges.append(p)

        return np.asarray(ranges)


    def pingCB(self, auv_ping, exp_ping, auv_pose, pf_pose):
        try:
            particle_tf = Transform()
            particle_tf.translation = auv_pose.pose.pose.position
            particle_tf.rotation    = auv_pose.pose.pose.orientation
            tf_mat = self.matrix_from_tf(particle_tf)
            m2auv = np.matmul(self.m2o_mat, np.matmul(tf_mat, self.base2mbes_mat))

            auv_ping_ranges = self.ping2ranges(auv_ping)
            exp_ping_ranges = self.pcloud2ranges(exp_ping, m2auv)
            #  print "------"
            #  print len(auv_ping_ranges)
            #  print len(exp_ping_ranges)

            # TODO: do the trimming of pings better than this
            self.waterfall.append(abs(auv_ping_ranges[:self.max_height]
                                      - exp_ping_ranges[:self.max_height]))
            self.active_auv_poses.append(auv_pose)
            beams_vec = self.ping2vecs(exp_ping, m2auv)
            self.active_pf_pings.append(beams_vec[:self.max_height])

            if len(self.waterfall)>self.max_height:
                self.waterfall.pop(0)
                self.active_auv_poses.pop(0)
                self.active_pf_pings.pop(0)

            self.new_msg = True

        except rospy.ROSInternalException:
            pass

    def matrix_from_tf(self, transform):
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
