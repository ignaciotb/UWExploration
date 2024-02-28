#!/usr/bin/env python3

# Python functionality
import sys

# ROS imports
import rospy
from nav_msgs.msg import Path

# Custom libraries
from PlannerClasses import SimplePlanner, BOPlanner


if __name__ == "__main__":
    choice = sys.argv[1]
    rospy.init_node("AUV_path_planner_node")
    corner_pub  = rospy.Publisher('/hugin_0/corners', Path, queue_size=1)
    path_pub    = rospy.Publisher('/hugin_0/waypoints', Path, queue_size=10)
    
    # Run lawnmower pattern
    if choice == "lawnmower":
        rospy.loginfo("Initializing planner node! Using Lawnmower pattern.")  
        planner = SimplePlanner('/hugin_0/corners', '/hugin_0/waypoints', [-250, -150, 0, -60], 8)
        corners = planner.generate_ip_corners()
        path = planner.generate_path(18, 0.2)
        corner_pub  = rospy.Publisher('/hugin_0/corners', Path, queue_size=1)
        path_pub    = rospy.Publisher('/hugin_0/waypoints', Path, queue_size=10)
        rospy.sleep(2)
        corner_pub.publish(corners)
        rospy.sleep(1)
        path_pub.publish(path) 
        rospy.loginfo("Published!")  
    
    # Run bayesian planner 
    if choice == "bo":
        rospy.loginfo("Initializing planner node! Using Bayesian Optimization.")  
        planner = BOPlanner('/hugin_0/corners', '/hugin_0/waypoints', [-250, -150, 0, -60], 8)
        corners = planner.generate_ip_corners()
        path = planner.initial_sampling_path(4)
        corner_pub  = rospy.Publisher('/hugin_0/corners', Path, queue_size=1)
        path_pub    = rospy.Publisher('/hugin_0/waypoints', Path, queue_size=10)
        rospy.sleep(2)
        corner_pub.publish(corners)
        rospy.sleep(1)
        path_pub.publish(path) 
        rospy.loginfo("Published first path!")
        
        # Wait until goal achieved.
            # Get 
            # Generate candidate waypoint. Publish
        

          
