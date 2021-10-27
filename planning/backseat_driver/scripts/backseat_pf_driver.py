#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
from move_base_msgs.msg import MoveBaseFeedback, MoveBaseResult, MoveBaseAction, MoveBaseGoal
import actionlib
import rospy
import tf
from std_msgs.msg import Float64, Header, Bool
import math
from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped, Point


class BackseatDriver(object):

    # create messages that are used to publish feedback/result
    _feedback = MoveBaseFeedback()
    _result = MoveBaseResult()

    def __init__(self, name):
        self._action_name = name

        #self.heading_offset = rospy.get_param('~heading_offsets', 5.)
        # self.planner_as_name = rospy.get_param('~path_planner_as')
        self.path_topic = rospy.get_param('~path_topic')
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.avg_pose_top = rospy.get_param("~average_pose_topic", '/average_pose')
        self.cov_threshold = rospy.get_param("~cov_threshold", 50)
        self.wp_topic = rospy.get_param('~wp_topic')
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 5.)

        # To send LC wp to the mission planner
        self.wp_pub = rospy.Publisher(self.wp_topic, PoseStamped, queue_size=1)

        self.listener = tf.TransformListener()

        # The waypoints as a path
        rospy.Subscriber(self.path_topic, Path, self.path_cb, queue_size=1)
        self.latest_path = Path()

        # The PF filter state
        rospy.Subscriber(self.avg_pose_top, PoseWithCovarianceStamped,
                         self.pf_cb, queue_size=1)
        self.closing_loop = False
        self.new_wp = PoseStamped()

        # The LC waypoints, as a path
        self.lc_waypoints = Path()
        self.lc_waypoints.header.frame_id = self.map_frame
        # One LC wp for testing
        lc_wp = PoseStamped()
        lc_wp.header.frame_id = self.map_frame
        lc_wp.pose.position.x = -183.65
        lc_wp.pose.position.y = -332.39
        lc_wp.pose.position.z = 0.
        lc_wp.pose.orientation.w = 1
        self.lc_waypoints.poses.append(lc_wp)

        rospy.spin()


    def path_cb(self, path_msg):
        self.latest_path = path_msg
        rospy.loginfo("Path received with number of wp: %d",
                      len(self.latest_path.poses))


    def pf_cb(self, pf_msg):
        # Reconstruct covariance 
        self.cov = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                self.cov[i, j] = pf_msg.pose.covariance[i*3 + j]

        # Reconstruct pose estimate
        position_estimate = pf_msg.pose.pose.position
        
        # Monitor trace
        trc = np.sum(np.diag(self.cov))
        if trc > self.cov_threshold and not self.closing_loop:
            # Pose uncertainty too high, closing the loop to relocalize
            # Find LC wp between current PF pose and next wp on the survey
            self.new_wp = self.lc_waypoints.poses[0]

            # Preempt current waypoint/path

            self.wp_pub.publish(self.new_wp)
            rospy.loginfo("Sent LC waypoint")
            self.closing_loop = True
        
        elif self.closing_loop:
            rospy.loginfo("Going for a loop closure!")

            try:
                (trans, rot) = self.listener.lookupTransform(
                    self.map_frame, self.base_frame, rospy.Time(0))
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                return

            start_pos = np.array(trans)
            end_pos = np.array(
                [self.new_wp.pose.position.x, self.new_wp.pose.position.y, self.new_wp.pose.position.z])

            rospy.loginfo("BS driver diff " + str(np.linalg.norm(start_pos - end_pos)))
            if np.linalg.norm(start_pos - end_pos) < self.goal_tolerance:
                # Goal reached
                rospy.loginfo("Loop closed!")
                self.closing_loop = False


if __name__ == '__main__':

    rospy.init_node('backseat_driver')
    bs_driver = BackseatDriver(rospy.get_name())
