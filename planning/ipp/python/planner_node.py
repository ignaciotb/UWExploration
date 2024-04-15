#!/usr/bin/env python3

# Python functionality
import sys

# ROS imports
import rospy
from nav_msgs.msg import Path

# Custom libraries
from PlannerClasses import SimplePlanner, BOPlanner
from GaussianProcessClass import SVGP_map
from BayesianOptimizerClass import BayesianOptimizer


if __name__ == "__main__":
    
    rospy.init_node("AUV_path_planner_node")
    try:
        choice = sys.argv[1]
        corner_pub  = rospy.Publisher('/hugin_0/corners', Path, queue_size=1)
        path_pub    = rospy.Publisher('/hugin_0/waypoints', Path, queue_size=10)
        #auv_ui_online()
        # Run lawnmower pattern
        #choice="lawnmower"
        if choice == "lawnmower":
            rospy.loginfo("Initializing planner node! Using Lawnmower pattern.")  
            planner = SimplePlanner('/hugin_0/corners', '/hugin_0/waypoints', [-270, -40, 130, -100], 8)
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
        else:
            rospy.loginfo("Initializing planner node! Using Bayesian Optimization.")  
            planner = BOPlanner('/hugin_0/corners', '/hugin_0/waypoints', [-250, 0, 100, -100], 8)
            # BO planner should have a SVGP up and running initially, that trains on all incoming samples
            # Wait until goal achieved.
                # Get 
                # Generate candidate waypoint. Publish
    except rospy.ROSInterruptException:
        rospy.logerr('Could not launch AUV path planner node')
        

          
