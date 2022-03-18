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


class W2WMissionPlanner(object):

    # create messages that are used to publish feedback/result
    _feedback = MoveBaseFeedback()
    _result = MoveBaseResult()

    def __init__(self, name):
        self._action_name = name

        #self.heading_offset = rospy.get_param('~heading_offsets', 5.)
        self.planner_as_name = rospy.get_param('~path_planner_as')
        self.path_topic = rospy.get_param('~path_topic')
        self.wp_topic = rospy.get_param('~wp_topic')
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.relocalize_topic = rospy.get_param('~relocalize_topic')

        # The waypoints as a path
        rospy.Subscriber(self.path_topic, Path, self.path_cb, queue_size=1)
        self.latest_path = Path()

        # LC waypoints, individually
        rospy.Subscriber(self.wp_topic, PoseStamped, self.wp_cb, queue_size=1)

        # The bs driver can be fairly slow on the simulations, so it's necessary
        # to stop the vehicle until the LC area has been selected
        rospy.Subscriber(self.relocalize_topic, Bool, self.start_relocalize, queue_size=1)
        self.relocalizing = False

        # The client to send each wp to the server
        self.ac = actionlib.SimpleActionClient(self.planner_as_name, MoveBaseAction)
        while not self.ac.wait_for_server(rospy.Duration(1)) and not rospy.is_shutdown():
            rospy.loginfo("Waiting for action client: %s",
                          self.planner_as_name)
        rospy.loginfo("Action client connected: %s", self.planner_as_name)


        while not rospy.is_shutdown():
            
            if self.latest_path.poses and not self.relocalizing:
                # Get next waypoint in path
                rospy.loginfo("Sending WP")
                wp = self.latest_path.poses[0]
                del self.latest_path.poses[0]

                # TODO: normalize quaternions here according to rviz warning?
                goal = MoveBaseGoal(wp)
                goal.target_pose.header.frame_id = self.map_frame
                self.ac.send_goal(goal)
                self.ac.wait_for_result()
                rospy.loginfo("WP reached, moving on to next one")

            elif not self.latest_path.poses:
                rospy.loginfo_once("Mission finished")
            

    def start_relocalize(self, bool_msg):
        self.relocalizing = bool_msg.data

    def path_cb(self, path_msg):
        self.latest_path = path_msg
        rospy.loginfo("Path received with number of wp: %d",
                      len(self.latest_path.poses))

    def wp_cb(self, wp_msg):
        # Waypoints for LC from the backseat driver
        rospy.loginfo("LC wp received")
        self.latest_path.poses.insert(0, wp_msg)


if __name__ == '__main__':

    rospy.init_node('w2w_mission_planner')
    planner = W2WMissionPlanner(rospy.get_name())
