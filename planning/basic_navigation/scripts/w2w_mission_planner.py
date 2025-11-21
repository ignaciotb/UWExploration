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
from ipp.msg import PathPlanAction, PathPlanGoal, PathPlanResult


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

        # AS for minibath training data from RBPF
        bo_replan_topic = rospy.get_param("~bo_replan_as")
        self.ac_plan = actionlib.SimpleActionClient(bo_replan_topic, PathPlanAction)
        while not self.ac_plan.wait_for_server(timeout=rospy.Duration(5)) and not rospy.is_shutdown():
                        rospy.loginfo("Waiting for IPP action server: %s", bo_replan_topic)
        rospy.loginfo("IPP action client connected: %s", bo_replan_topic)
        self.replanning = False

        # The client to send each wp to the controller server
        self.ac = actionlib.SimpleActionClient(self.planner_as_name, MoveBaseAction)
        while not self.ac.wait_for_server(rospy.Duration(1)) and not rospy.is_shutdown():
            rospy.loginfo("Waiting for action client: %s",
                          self.planner_as_name)
        rospy.loginfo("Action client connected: %s", self.planner_as_name)

        rospy.loginfo("Waiting for synch service")
        synch_top = rospy.get_param("~synch_topic", '/pf_synch')
        rospy.wait_for_service(synch_top)
        rospy.loginfo("Synch service started")

        # Ask IPP for initial path
        rate = rospy.Rate(1)
        rospy.loginfo("Calling IPP planner for initial path")
        self.request_ipp_path(0)
        self.ac_plan.wait_for_result()
        result = self.ac_plan.get_result()
        while len(result.path.poses) <= 0 and not rospy.is_shutdown():
            self.request_ipp_path(0)
            self.ac_plan.wait_for_result()
            result = self.ac_plan.get_result()
            rate.sleep() 

        self.latest_path.poses += result.path.poses
        rospy.loginfo("Path received with number of wp: %d",
                                len(result.path.poses))
        
        while not rospy.is_shutdown():            
            if self.latest_path.poses:
                # Get next waypoint in path
                wp = self.latest_path.poses[0]
                del self.latest_path.poses[0]

                # TODO: normalize quaternions here according to rviz warning?
                goal = MoveBaseGoal(wp)
                goal.target_pose.header.frame_id = self.map_frame
                self.ac.send_goal(goal)
                self.ac.wait_for_result()
                rospy.loginfo("WP reached, moving on to next one %d",
                                len(self.latest_path.poses))

                # Request new IPP path when only 3 wp left. 
                # TODO: this needs to be done based on time to last wp
                if len(self.latest_path.poses) < 3 and not self.replanning:
                    rospy.loginfo("Reaching final WP. Calling IPP planner")
                    self.request_ipp_path(2)
                    self.replanning = True
                    
                if self.replanning:
                    if self.ac_plan.wait_for_result(rospy.Duration(0.01)):
                        result = self.ac_plan.get_result()
                        self.latest_path.poses += result.path.poses
                        rospy.loginfo("Path received with number of wp: %d",
                                    len(result.path.poses))
                        self.replanning = False                    
                                    
            elif not self.latest_path.poses:
                rospy.loginfo_once("Mission finished")

            rate.sleep()
            

    def request_ipp_path(self, type):
        goal = PathPlanGoal()
        goal.request = type
        self.ac_plan.send_goal(goal)

        # return result

    def path_cb(self, path_msg):
        self.latest_path = path_msg
        rospy.loginfo("Path received with number of wp: %d",
                      len(self.latest_path.poses))


if __name__ == '__main__':

    rospy.init_node('w2w_mission_planner')
    planner = W2WMissionPlanner(rospy.get_name())
