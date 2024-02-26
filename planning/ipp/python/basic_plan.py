#!/usr/bin/env python3

import numpy as np
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import std_msgs.msg

"""
This script initializes a rectangle shape with corners,
and passes these to generate inducing points for the SVGP. 
Then it also creates a set of waypoints inside this shape 
in a lawnmower pattern, and passes this to the navigation stack.
"""

def generate_paths(sW, sO, tR, fM = False, l = -280, r = -200, u = 100, d = 60):
    
    #ul: -307, 118
    #ur: -37, 153
    #dl: -266, -102
    #dr: 13, -60

    H = u - d
    W = r - l
    alpha = max((1-sO)*sW, 1.1*tR)
    if fM:
        beta = 0
    else:
        beta = tR
    
    n = round(H/alpha)

    h = std_msgs.msg.Header()
    h.stamp = rospy.Time.now()
    corners = Path()
    corners.header = h
    lm_path = Path()
    lm_path.header = h
    
    # Append corners
    ul_c = PoseStamped()
    ul_c.pose.position.x = l
    ul_c.pose.position.y = u
    ul_c.header = h
    corners.poses.append(ul_c)
    

    ur_c = PoseStamped()
    ur_c.header = h
    ur_c.pose.position.x = r
    ur_c.pose.position.y = u
    corners.poses.append(ur_c)

    dl_c = PoseStamped()
    dl_c.header = h
    dl_c.pose.position.x = l
    dl_c.pose.position.y = d
    corners.poses.append(dl_c)

    dr_c = PoseStamped()
    dr_c.header = h
    dr_c.pose.position.x = r
    dr_c.pose.position.y = d
    corners.poses.append(dr_c)

    x = l + beta
    y = u + alpha
    for i in range(n):
        if i % 2 == 0:
            h_1 = std_msgs.msg.Header()
            h_1.stamp = rospy.Time.now()
            wp1 = PoseStamped()
            wp1.header = h_1
            wp1.pose.position.x = x
            y = y - alpha
            wp1.pose.position.y = y
            lm_path.poses.append(wp1)
            h_2 = std_msgs.msg.Header()
            h_2.stamp = rospy.Time.now()
            wp2 = PoseStamped()
            wp2.header = h_2
            x = x + W - 2*beta
            wp2.pose.position.x = x
            wp2.pose.position.y = y
            lm_path.poses.append(wp2)
        else:
            h_1 = std_msgs.msg.Header()
            h_1.stamp = rospy.Time.now()
            wp1 = PoseStamped()
            wp1.header = h_1
            wp1.pose.position.x = x
            y = y - alpha
            wp1.pose.position.y = y
            lm_path.poses.append(wp1)
            h_2 = std_msgs.msg.Header()
            h_2.stamp = rospy.Time.now()
            wp2 = PoseStamped()
            wp2.header = h_2
            x = x - W + 2*beta
            wp2.pose.position.x = x
            wp2.pose.position.y = y
            lm_path.poses.append(wp2)

    return corners, lm_path

if __name__ == "__main__":
    rospy.init_node("lawnmower_planner")
    corners, lawnmower_path = generate_paths(18, 0, 8)
    corner_pub  = rospy.Publisher('/hugin_0/corners', Path, queue_size=1)
    path_pub    = rospy.Publisher('/hugin_0/waypoints', Path, queue_size=10)
    rospy.sleep(2)
    corner_pub.publish(corners)
    path_pub.publish(lawnmower_path) 
    rospy.loginfo("Published!")   
