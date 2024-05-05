#!/usr/bin/env python3

# Python functionality
import sys

# ROS imports
import rospy

# Custom libraries
from PlannerClasses import SimplePlanner, BOPlanner


if __name__ == "__main__":
    
    rospy.init_node("AUV_path_planner_node")
    
    # Get parameters from ROS
    choice              = rospy.get_param("~planner_type")
    turn_radius         = rospy.get_param("~turning_radius")
    corner_topic        = rospy.get_param("~corner_topic")
    path_topic          = rospy.get_param("~path_topic")
    planner_req_topic   = rospy.get_param("~planner_req_topic")
    odom_topic          = rospy.get_param("~odom_topic")
    swath_width         = rospy.get_param("~swath_width")
    swath_overlap       = rospy.get_param("~swath_overlap")
    bound_left          = rospy.get_param("~bound_left")
    bound_right         = rospy.get_param("~bound_right")
    bound_up            = rospy.get_param("~bound_up")
    bound_down          = rospy.get_param("~bound_down")
    train_rate          = rospy.get_param("~train_rate")
    wp_resolution       = rospy.get_param("~wp_resolution")
    path_nbr_samples    = rospy.get_param("~path_nbr_samples")
    voxel_size          = rospy.get_param("~voxel_size")
    wp_sample_interval  = rospy.get_param("~wp_sample_interval")
    horizon_distance    = rospy.get_param("~horizon_distance")
    border_margin       = rospy.get_param("~border_margin")
    beta                = rospy.get_param("~beta")
    
    bounds = [bound_left, bound_right, bound_up, bound_down]
    
    try:        
        # Run lawnmower pattern
        if choice == "lawnmower":
            rospy.loginfo("Initializing planner node! Using Lawnmower pattern.")  
            planner = SimplePlanner(corner_topic=corner_topic, path_topic=path_topic, 
                                    planner_req_topic=planner_req_topic, odom_topic=odom_topic,
                                    bounds=bounds, 
                                    turning_radius=turn_radius, training_rate=train_rate, sw=swath_width, 
                                    so=swath_overlap)
        
        # Run bayesian optimization based planner 
        else:
            rospy.loginfo("Initializing planner node! Using Bayesian Optimization.")  
            planner = BOPlanner(corner_topic=corner_topic, path_topic=path_topic, planner_req_topic=planner_req_topic, 
                                odom_topic=odom_topic,bounds=bounds, turning_radius=turn_radius, training_rate=train_rate, 
                                wp_resolution=wp_resolution, swath_width=swath_width, path_nbr_samples=path_nbr_samples, 
                                voxel_size=voxel_size, wp_sample_interval=wp_sample_interval, horizon_distance=horizon_distance,
                                border_margin=border_margin, beta=beta)
            
    except rospy.ROSInterruptException:
        rospy.logerr('Could not launch AUV path planner node')
        

          
