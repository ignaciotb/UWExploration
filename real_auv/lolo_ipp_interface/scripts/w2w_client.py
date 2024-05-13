#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
from move_base_msgs.msg import MoveBaseFeedback, MoveBaseResult, MoveBaseAction, MoveBaseGoal
from smarc_bt.msg import GotoWaypointAction, GotoWaypointGoal, GotoWaypoint, MissionControl
import actionlib
import rospy
import tf
from std_msgs.msg import Float64, Header, Bool
import math


class W2WClientLolo(object):

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
        # self.relocalize_topic = rospy.get_param('~relocalize_topic')
        self.planner_req_topic = rospy.get_param('~planner_req_topic')

        # The waypoints as a path
        rospy.Subscriber(self.path_topic, Path, self.path_cb, queue_size=1)
        self.latest_path = Path()
        self.first_recieved = False

        # LC waypoints, individually
        rospy.Subscriber(self.wp_topic, PoseStamped, self.wp_cb, queue_size=1)

        self.started = False
        self.planner_req_pub = rospy.Publisher(self.planner_req_topic, Bool)

        # The client to send each wp to the server
        self.ac = actionlib.SimpleActionClient(self.planner_as_name, GotoWaypointAction)
        # self.ac = actionlib.SimpleActionClient(self.planner_as_name, MoveBaseAction)
        while not self.ac.wait_for_server(rospy.Duration(1)) and not rospy.is_shutdown():
            rospy.loginfo("Waiting for action server: %s",
                          self.planner_as_name)
        rospy.loginfo("Action client connected: %s", self.planner_as_name)

        rospy.loginfo("Waiting for synch service")
        synch_top = rospy.get_param("~synch_topic", '/pf_synch')
        rospy.wait_for_service(synch_top)
        rospy.loginfo("Synch service started")
                

        while not rospy.is_shutdown():
            
            if self.latest_path.poses:
                # Get next waypoint in path
                rospy.loginfo("Sending WP")
                wp = self.latest_path.poses[0]
                del self.latest_path.poses[0]

                # TODO: normalize quaternions here according to rviz warning?
                goal = GotoWaypointGoal()
                # goal.target_pose.header.frame_id = self.map_frame
                print("Next WP: ", wp.pose.position.x, wp.pose.position.y)
                
                goal.waypoint.pose.pose.position.x = wp.pose.position.x
                goal.waypoint.pose.pose.position.y = wp.pose.position.y
                goal.waypoint.pose.header.frame_id = self.map_frame
                goal.waypoint.travel_depth = 0.
                goal.waypoint.travel_rpm = 250
                goal.waypoint.goal_tolerance = 10
                goal.waypoint.speed_control_mode = GotoWaypoint.SPEED_CONTROL_RPM
                goal.waypoint.z_control_mode = GotoWaypoint.Z_CONTROL_DEPTH
                # goal.waypoint.travel_altitude = altitude
                # goal.waypoint.z_control_mode = GotoWaypoint.Z_CONTROL_ALTITUDE

                self.ac.send_goal(goal)
                self.ac.wait_for_result()
                rospy.loginfo("WP reached, moving on to next one")
                self.started = True
                
                #if len(self.latest_path.poses) == 5:
                self.planner_req_pub.publish(True)
                
                        
            elif not self.latest_path.poses:
               rospy.loginfo_once("Mission finished")
            #    if self.needs_plan:
            #        self.planner_req_pub.publish(True)
            #        self.needs_plan = False
            #        self.started = False
                    
                #if self.started == True:
                    #self.planner_req_pub.publish(True)
                    #self.started = False
                
    def path_cb(self, path_msg):
        if not self.first_recieved:
            self.latest_path = path_msg
            rospy.loginfo("Adding new path with number of wp: %d",
                      len(path_msg.poses))
            self.first_recieved = True
        else:
            self.latest_path.poses.extend(path_msg.poses)
            rospy.loginfo("Appending new path with number of wp: %d",
                      len(path_msg.poses))

    def wp_cb(self, wp_msg):
        # Waypoints for LC from the backseat driver
        rospy.loginfo("LC wp received")
        self.latest_path.poses.insert(0, wp_msg)


if __name__ == '__main__':

    rospy.init_node('lolo_w2w_client')
    planner = W2WClientLolo(rospy.get_name())
