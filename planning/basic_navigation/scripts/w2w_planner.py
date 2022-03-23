#!/usr/bin/env python3

from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
from move_base_msgs.msg import MoveBaseFeedback, MoveBaseResult, MoveBaseAction
import actionlib
import rospy
import tf
from std_msgs.msg import Float64, Header, Bool
import math


class W2WPathPlanner(object):

    # create messages that are used to publish feedback/result
    _feedback = MoveBaseFeedback()
    _result = MoveBaseResult()

    def execute_cb(self, goal):

        rospy.loginfo("Goal received")

        success = True
        self.nav_goal = goal.target_pose.pose
        self.nav_goal_frame = goal.target_pose.header.frame_id
        if self.nav_goal_frame is None or self.nav_goal_frame == '':
            rospy.logwarn("Goal has no frame id! Using map by default")
            self.nav_goal_frame = self.map_frame

        r = rospy.Rate(10.)  # 10hz
        counter = 0
        while not rospy.is_shutdown() and self.nav_goal is not None:

            # Preempted
            if self._as.is_preempt_requested():
                rospy.loginfo('%s: Preempted' % self._action_name)
                self._as.set_preempted()
                self.nav_goal = None

                # Stop thruster
                self.motion_command(0., 0., 0.)
                break

            # Transform goal map --> base frame
            goal_point = PointStamped()
            goal_point.header.frame_id = self.nav_goal_frame
            goal_point.header.stamp = rospy.Time(0)
            goal_point.point.x = self.nav_goal.position.x
            goal_point.point.y = self.nav_goal.position.y
            goal_point.point.z = self.nav_goal.position.z
            try:
                goal_point_local = self.listener.transformPoint(
                    self.base_frame, goal_point)
                
                #Compute throttle error
                # throttle_level = min(self.max_throttle, np.linalg.norm(
                #     np.array([goal_point_local.point.x + goal_point_local.point.y])))
                # Nacho: no real need to adjust the throttle 
                throttle_level = self.max_throttle
                # Compute thrust error
                alpha = math.atan2(goal_point_local.point.y,
                                goal_point_local.point.x)
                sign = np.copysign(1, alpha)
                yaw_setpoint = sign * min(self.max_thrust, abs(alpha))

                # Command velocities
                self.motion_command(throttle_level, yaw_setpoint, 0.)

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                rospy.logwarn("Transform to base frame not available yet")
            pass

            # Publish feedback
            if counter % 10 == 0:
                self._as.publish_feedback(self._feedback)

            counter += 1
            r.sleep()

        # Stop thruster
        self.motion_command(0.,0.,0.)
        rospy.loginfo('%s: Succeeded' % self._action_name)
        self._as.set_succeeded(self._result)

    def motion_command(self, throttle_level, thruster_angle, inclination_angle):
        
        incl = Float64()
        throttle = Float64()
        thrust = Float64()

        throttle.data = throttle_level
        thrust.data = thruster_angle
        incl.data = inclination_angle
        self.thruster_pub.publish(thrust)
        self.inclination_pub.publish(incl)
        self.throttle_pub.publish(throttle)


    def timer_callback(self, event):
        if self.nav_goal is None:
            #rospy.loginfo_throttle(30, "Nav goal is None!")
            return

        # Check if the goal has been reached
        try:
            (trans, rot) = self.listener.lookupTransform(
                self.nav_goal_frame, self.base_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            return

        start_pos = np.array(trans)
        end_pos = np.array(
            [self.nav_goal.position.x, self.nav_goal.position.y, self.nav_goal.position.z])

        rospy.logdebug("diff " + str(np.linalg.norm(start_pos - end_pos)))
        if np.linalg.norm(start_pos - end_pos) < self.goal_tolerance:
            # Goal reached
            self.nav_goal = None


    def __init__(self, name):
        self._action_name = name

        self.goal_tolerance = rospy.get_param('~goal_tolerance', 5.)
        self.max_throttle = rospy.get_param('~max_throttle', 2.)
        self.max_thrust = rospy.get_param('~max_thrust', 0.5)
        self.map_frame = rospy.get_param('~map_frame', 'map') 
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.throttle_top = rospy.get_param('~throttle_cmd', '/throttle')
        self.thruster_top = rospy.get_param('~thruster_cmd', '/thruster')
        self.inclination_top = rospy.get_param('~inclination_cmd', '/inclination')
        self.as_name = rospy.get_param('~path_planner_as', 'path_planner')

        self.nav_goal = None

        self.listener = tf.TransformListener()
        rospy.Timer(rospy.Duration(2), self.timer_callback)

        self.throttle_pub = rospy.Publisher(self.throttle_top, Float64, queue_size=1)
        self.thruster_pub = rospy.Publisher(self.thruster_top, Float64, queue_size=1)
        self.inclination_pub = rospy.Publisher(self.inclination_top, Float64, queue_size=1)

        self._as = actionlib.SimpleActionServer(
            self.as_name, MoveBaseAction, execute_cb=self.execute_cb, auto_start=False)
        self._as.start()
        rospy.loginfo("Announced action server with name: %s", self.as_name)

        rospy.spin()


if __name__ == '__main__':

    rospy.init_node('w2w_path_planner')
    planner = W2WPathPlanner(rospy.get_name())
