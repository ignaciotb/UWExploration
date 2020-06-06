#!/usr/bin/python

# Standard dependencies
import sys
import os
import math
import rospy
import numpy as np
import tf
import tf2_ros
from scipy.special import logsumexp # For log weights

from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion, TransformStamped, Vector3
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf.transformations import translation_matrix, translation_from_matrix
from tf.transformations import quaternion_matrix, quaternion_from_matrix


# For sim mbes action client
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

# Import Particle() class
from auv_particle import Particle, matrix_from_tf, pcloud2ranges

# Multiprocessing and parallelizing
import numba
from numba import jit
from resampling import residual_resample, naive_resample, systematic_resample, stratified_resample

# For sim mbes action client
import actionlib
from auv_2_ros.msg import MbesSimGoal, MbesSimAction

#Profiling
# import cProfile

# import time # For evaluating mp improvements
# import multiprocessing as mp
# from functools import partial # Might be useful with mp
# from pathos.multiprocessing import ProcessingPool as Pool

class auv_pf(object):

    def __init__(self):
        # Read necessary parameters
        self.pc = rospy.get_param('~particle_count', 10) # Particle Count
        self.map_frame = rospy.get_param('~map_frame', 'map') # map frame_id
        odom_frame = rospy.get_param('~odom_frame', 'odom')
        meas_model_as = rospy.get_param('~mbes_as', '/mbes_sim_server') # map frame_id
        mbes_pc_top = rospy.get_param("~particle_sim_mbes_topic", '/sim_mbes')
        self.markers  = rospy.get_param("~markers", "true")

        if self.markers:
            marker_topic =  rospy.get_param("~marker_topic", '/pf/particle_markers')
            marker_model = rospy.get_param("~marker_model", 'package://hugin_description/mesh/hugin_color.dae')
            self.markers_list = MarkerArray()
            vector = Vector3()
            vector.x = 0.001
            vector.y = 0.001
            vector.z = 0.001
            markers_p = []
        else:
            self.poses = PoseArray()
        self.pose_list = np.zeros((self.pc,6)) #x,y,z,roll,pitch,yaw
        # Initialize tf listener
        tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tfBuffer)
        try:
            rospy.loginfo("Waiting for transforms")
            mbes_tf = tfBuffer.lookup_transform('hugin/base_link', 'hugin/mbes_link',
                                                rospy.Time(0), rospy.Duration(10))
            mbes2base_mat = matrix_from_tf(mbes_tf)

            m2o_tf = tfBuffer.lookup_transform('map', 'odom', rospy.Time(0), rospy.Duration(10))
            m2o_mat = matrix_from_tf(m2o_tf)

            rospy.loginfo("Transforms locked - pf node")
        except:
            rospy.loginfo("ERROR: Could not lookup transform from base_link to mbes_link")

        # Read covariance values
        meas_cov = float(rospy.get_param('~measurement_covariance', 0.01))
        cov_string = rospy.get_param('~motion_covariance')
        cov_string = cov_string.replace('[','')
        cov_string = cov_string.replace(']','')
        cov_list = list(cov_string.split(", "))
        motion_cov = list(map(float, cov_list))

        cov_string = rospy.get_param('~init_covariance')
        cov_string = cov_string.replace('[','')
        cov_string = cov_string.replace(']','')
        cov_list = list(cov_string.split(", "))
        init_cov = list(map(float, cov_list))


        # Initialize list of particles
        self.particles = np.empty(self.pc, dtype=object)
        self.ac = np.empty(self.pc, dtype=object)

        for i in range(self.pc):
            self.particles[i] = Particle(i, self.pc, mbes2base_mat, m2o_mat, init_cov=init_cov, meas_cov=meas_cov,
                                     process_cov=motion_cov, map_frame=self.map_frame, odom_frame=odom_frame, pc_mbes_top=mbes_pc_top)

            self.ac[i] = actionlib.SimpleActionClient(meas_model_as, MbesSimAction)       # Initialize connection to MbesSim action server
            self.ac[i].wait_for_server()


        self.time = None
        self.old_time = None
        self.pred_odom = None
        self.latest_mbes = PointCloud2()
        self.prev_mbes = PointCloud2()
        self.weights = np.zeros((70,))

        for i in range(self.pc):
            if self.markers:
                marker = Marker()
                marker.header.frame_id = odom_frame
                marker.pose = self.particles[i].p_pose
                marker.ns = "hugin"
                marker.id = i
                marker.scale.x = 1.0
                marker.scale.y = 1.0
                marker.scale.z = 1.0
                # marker.color.r = 1.0
                # marker.color.g = 0.0
                marker.color.b = 1.0
                marker.color.a = 1.0
                marker.scale = vector
                marker.type = 10
                marker.mesh_resource = marker_model
                markers_p.append(marker)

                self.markers_list.markers = markers_p
            else:
                self.poses.poses.append(self.particles[i].p_pose)
            self.pose_list[i] = self.particles[i].update_pose_vec()

        if not self.markers:
            self.poses.header.frame_id = odom_frame

        self.avg_pose = PoseWithCovarianceStamped()
        self.avg_pose.header.frame_id = odom_frame

        # Initialize particle poses publisher
        if self.markers:
            self.pf_pub = rospy.Publisher(marker_topic, MarkerArray, queue_size=10)
        else:
            pose_array_top = rospy.get_param("~particle_poses_topic", '/particle_poses')
            self.pf_pub = rospy.Publisher(pose_array_top, PoseArray, queue_size=10)

        # Initialize average of poses publisher
        avg_pose_top = rospy.get_param("~average_pose_topic", '/average_pose')
        self.avg_pub = rospy.Publisher(avg_pose_top, PoseWithCovarianceStamped, queue_size=10)

        # Establish subscription to mbes pings message
        mbes_pings_top = rospy.get_param("~mbes_pings_topic", 'mbes_pings')
        rospy.Subscriber(mbes_pings_top, PointCloud2, self.mbes_callback)

        # Establish subscription to odometry message (intentionally last)
        odom_top = rospy.get_param("~odometry_topic", 'odom')
        rospy.Subscriber(odom_top, Odometry, self.odom_callback)

        rospy.loginfo("Particle filter class successfully created")

        self.update_rviz()
        rospy.spin()

    def mbes_callback(self, msg):
        self.latest_mbes = msg

    def odom_callback(self, odom_msg):
        self.time = odom_msg.header.stamp.to_sec()
        if self.old_time and self.time > self.old_time:
            # Motion prediction
            self.predict(odom_msg)

            if self.latest_mbes.header.stamp > self.prev_mbes.header.stamp:
                # Measurement update if new one received
                self.update(self.latest_mbes, odom_msg)
                self.prev_mbes = self.latest_mbes

                # Particle resampling
                self.resample()

            self.update_rviz()
        self.old_time = self.time

    def predict(self, odom_t):
        #  dt = 0.1
        dt = self.time - self.old_time
        for i in range(0, self.pc):
            self.particles[i].motion_pred(odom_t, dt)

    def update(self, meas_mbes, odom):
        mbes_meas_ranges = pcloud2ranges(meas_mbes, odom.pose.pose)


        for i in range(self.pc):
            mbes_goal = self.particles[i].meas_update(mbes_meas_ranges)
            self.ac[i].send_goal(mbes_goal)

        for i in range(self.pc):
            self.ac[i].wait_for_result()
            mbes_res = self.ac[i].get_result()

            # Pack result into PointCloud2
            mbes_pcloud = PointCloud2()
            mbes_pcloud = mbes_res.sim_mbes
            mbes_pcloud.header.frame_id = self.map_frame

            self.particles[i].update_weight(mbes_pcloud, mbes_meas_ranges)
            self.weights[i] = self.particles[i].w

        # Add small non-zero value to avoid hitting zero
        self.weights += 1.e-30

        return

    def resample(self):

        print "-------------"
        # Normalize weights
        self.weights /= self.weights.sum()
        #  print "Weights"
        #  print weights

        N_eff = self.pc
        if self.weights.sum() == 0.:
            rospy.loginfo("All weights zero!")
        else:
            N_eff = 1/np.sum(np.square(self.weights))

        print "N_eff ", N_eff
        # Resampling?
        if N_eff < self.pc*0.5:
            indices = residual_resample(self.weights)
            print "Indices"
            print indices
            keep = list(set(indices))
            lost = [i for i in range(self.pc) if i not in keep]
            dupes = indices[:].tolist()
            for i in keep:
                dupes.remove(i)

            self.reassign_poses(lost, dupes)

            # Add noise to particles
            for i in range(self.pc):
                self.particles[i].add_noise([3.,3.,0.,0.,0.,0.0])

        else:
            print N_eff
            rospy.loginfo('Number of effective particles high - not resampling')

    def reassign_poses(self, lost, dupes):
        for i in range(len(lost)):
            # Faster to do separately than using deepcopy()
            self.particles[lost[i]].p_pose.position.x = self.particles[dupes[i]].p_pose.position.x
            self.particles[lost[i]].p_pose.position.y = self.particles[dupes[i]].p_pose.position.y
            self.particles[lost[i]].p_pose.position.z = self.particles[dupes[i]].p_pose.position.z
            self.particles[lost[i]].p_pose.orientation.x = self.particles[dupes[i]].p_pose.orientation.x
            self.particles[lost[i]].p_pose.orientation.y = self.particles[dupes[i]].p_pose.orientation.y
            self.particles[lost[i]].p_pose.orientation.z = self.particles[dupes[i]].p_pose.orientation.z
            self.particles[lost[i]].p_pose.orientation.w = self.particles[dupes[i]].p_pose.orientation.w

    def average_pose(self, poses_array):
        """
        Get average pose of particles and
        publish it as PoseWithCovarianceStamped

        :param pose_list: List of lists containing pose
                        of all particles in form
                        [x, y, z, roll, pitch, yaw]
        :type pose_list: list
            """
        ave_pose = poses_array.mean(axis = 0)

        self.avg_pose.pose.pose.position.x = ave_pose[0]
        self.avg_pose.pose.pose.position.y = ave_pose[1]
        """
        If z, roll, and pitch can stay as read directly from
        the odometry message there is no need to average them.
        We could just read from any arbitrary particle
        """
        self.avg_pose.pose.pose.position.z = ave_pose[2]
        roll  = ave_pose[3]
        pitch = ave_pose[4]
        """
        Average of yaw angles creates
        issues when heading towards pi because pi and
        negative pi are next to eachother, but average
        out to zero (opposite direction of heading)
        """
        yaws = poses_array[:,5]
        # print(yaws)
        # yaws = np.where(np.abs(yaws) > 2*np.pi, yaws + 2*np.pi, yaws)
        #Something feels wrong with the code below, need to think more on it
        if np.abs(yaws).min() > math.pi/2:
            yaws[yaws < 0] += 2*math.pi
        yaw = yaws.mean()

        self.avg_pose.pose.pose.orientation = Quaternion(*quaternion_from_euler(roll, pitch, yaw))
        self.avg_pose.header.stamp = rospy.Time.now()
        self.avg_pub.publish(self.avg_pose)


    # TODO: publish markers instead of poses
    #       Optimize this function
    def update_rviz(self):

        for i in range(self.pc):
            if self.markers:
                self.markers_list.markers[i].pose = self.particles[i].p_pose
                self.markers_list.markers[i].header.stamp = rospy.Time.now()
            else:
                self.poses.poses[i] = self.particles[i].p_pose
            self.particles[i].update_pose_vec()
            self.pose_list[i] = self.particles[i].pose_vec
        # Publish particles with time odometry was received
        #self.poses.header.stamp = rospy.Time.now()
        if self.markers:
            self.pf_pub.publish(self.markers_list)
        else:
            self.pf_pub.publish(self.poses)

        self.average_pose(self.pose_list)


if __name__ == '__main__':

    rospy.init_node('auv_pf')
    try:
        auv_pf()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch pf")
        pass
