# Torch libraries
import torch
import botorch

# Math libraries
import numpy as np
import math

# ROS imports
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import std_msgs.msg
import tf.transformations
import tf
import tf_conversions
import geometry_msgs

# Threading safety support
import filelock

# Python functionality
import filelock
import copy

# Custom imports
import PlannerTemplateClass
import ipp_utils

class LMPlanner(PlannerTemplateClass.PlannerTemplate):
    """ Class for lawnmower pattern planner.
        This class will publish an the full path in a lawnmower
        pattern as soon as it is instanciated. When reaching
        waypoints, it will save the trained environment model.
    
        This class implements the following functions:
        
        `update_wp_cb`
        `generate_path`
    
    Args:
        PlannerBase (obj): Basic template of planner class
    """
    def __init__(self, corner_topic, path_topic, planner_req_topic, odom_topic, 
                 bounds, turning_radius, training_rate, sw):
        """ Constructor

        Args:
            corner_topic (string): _description_
            path_topic (string): _description_
            planner_req_topic (string): _description_
            odom_topic (string): _description_
            bounds (list(float)): _description_
            turning_radius (double): _description_
            training_rate (int): _description_
            sw (double): _description_
        """
        # Invoke constructor of parent class
        super().__init__(corner_topic, path_topic, planner_req_topic, odom_topic, 
                         bounds, turning_radius, training_rate) 
        
        # Publish path, then train GP
        self.path_pub = rospy.Publisher(path_topic, Path, queue_size=100)
        rospy.sleep(1)
        path = self.generate_path(swath_width=sw)
        self.path_pub.publish(path) 
        self.begin_gp_train()
            
    def update_wp_cb(self, msg):
        """ When called, dumps GP

        Args:
            msg (bool): dummy boolean, not used currently
        """
        
        # Freeze a copy of current model for plotting, to let real model keep training
        with self.gp.mutex:
            ipp_utils.save_model(self.gp.model, self.store_path + "_GP_" + str(round(self.distance_travelled)) + "_env_lawnmower.pickle")
            print("Saved model")
        
        # Notify of current distance travelled
        print("Current distance travelled: " + str(round(self.distance_travelled)) + " m.")
        
    
    def generate_path(self, swath_width):
        """ Creates a basic path in lawnmower pattern in a rectangle with given parameters

        Args:
            swath_width     (double): Swath width of MBES sensor
            max_time        (double): Time the trajectory is allowed to take (seconds)
            vehicle_speed   (double): speed of vehicle (m/s)
            
        Returns:
            nav_msgs.msg.Path: Waypoint list, in form of poses
        """
        
        while not self.odom_init and not rospy.is_shutdown():
            print("Lawnmower patternis waiting for odometry before starting.")
            rospy.sleep(2)
            
        swath_width = 30
        turning_radius = 15
        
        low_x   = self.bounds[0]
        low_y   = self.bounds[1]
        high_x  = self.bounds[2]
        high_y  = self.bounds[3]
        
        print(self.bounds)
        
        # Check which corner to start on, based on which is closest
        if abs(self.state[0] - low_x) < abs(self.state[0] - high_x):
            start_x = low_x
            direction_x = 1
        else:
            start_x = high_x
            direction_x = -1
            
        if abs(self.state[1] - low_y) < abs(self.state[1] - high_y):
            start_y = low_y
            direction_y = 1
        else:
            start_y = high_y
            direction_y = -1
            
        # Calculate how many passes, floor to be safe and get int
        height = abs(high_y - low_y)
        width = abs(high_x - low_x)
        nbr_passes = math.ceil(width/max(swath_width, turning_radius))      
            
        # Calculate changes at each pass. Use Y as long end.
        dx = max(swath_width, 2*turning_radius) * direction_x
        dy = abs(height-2*turning_radius) * direction_y
        
        # Get stamp
        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()
        h.frame_id = self.map_frame
        lm_path = Path(header=h)
        
        # Calculate starting position
        h.stamp = rospy.Time.now()
        wp1 = PoseStamped(header=h)
        wp1.pose.position.x = start_x + direction_x * swath_width / 2
        wp1.pose.position.y = start_y + direction_x * turning_radius
        #lm_path.poses.append(wp1)
        start_x = start_x - direction_x * swath_width / 2
        start_y = start_y + direction_y * turning_radius
        x = start_x 
        y = start_y

        # Iterate to append waypoints to path
        for i in range(nbr_passes):
            if i % 2 == 0:
                h.stamp = rospy.Time.now()
                wp1 = PoseStamped(header=h)
                x = x + dx
                wp1.pose.position.x = x
                wp1.pose.position.y = y
                lm_path.poses.append(wp1)
                h.stamp = rospy.Time.now()
                wp2 = PoseStamped(header=h)
                y = y + dy
                wp2.pose.position.x = x
                wp2.pose.position.y = y
                lm_path.poses.append(wp2)
            else:
                h.stamp = rospy.Time.now()
                wp1 = PoseStamped(header=h)
                x = x + dx
                wp1.pose.position.x = x
                wp1.pose.position.y = y
                lm_path.poses.append(wp1)
                h.stamp = rospy.Time.now()
                wp2 = PoseStamped(header=h)
                y = y - dy
                wp2.pose.position.x = x
                wp2.pose.position.y = y
                lm_path.poses.append(wp2)

        return lm_path 