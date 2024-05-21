# Torch libraries
import torch
import botorch

# Math libraries
import numpy as np

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
                 bounds, turning_radius, training_rate, sw, max_time, vehicle_velocity):
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
                         bounds, turning_radius, training_rate, max_time, vehicle_velocity) 
        
        # Publish path, then train GP
        self.path_pub = rospy.Publisher(path_topic, Path, queue_size=100)
        rospy.sleep(1)
        path = self.generate_path(sw, self.max_time, self.vehicle_velocity)
        self.path_pub.publish(path) 
        self.begin_gp_train()
            
    def update_wp_cb(self, msg):
        """ When called, dumps GP

        Args:
            msg (bool): dummy boolean, not used currently
        """
        
        # Freeze a copy of current model for plotting, to let real model keep training
        with self.gp.mutex:
            pickle.dump(self.gp.model, open(self.store_path + "_GP_" + str(round(self.distance_travelled)) + "_env_lawnmower.pickle" , "wb"))
            print("pickled")
        
        # Notify of current distance travelled
        print("Current distance travelled: " + str(round(self.distance_travelled)) + " m.")
        
    
    def generate_path(self, swath_width, max_time, vehicle_speed):
        """ Creates a basic path in lawnmower pattern in a rectangle with given parameters

        Args:
            swath_width     (double): Swath width of MBES sensor
            max_time        (double): Time the trajectory is allowed to take (seconds)
            vehicle_speed   (double): speed of vehicle (m/s)
            
        Returns:
            nav_msgs.msg.Path: Waypoint list, in form of poses
        """
        
        l = self.bounds[0]
        r = self.bounds[1]
        u = self.bounds[2]
        d = self.bounds[3]
        
        # Check which corner to start on, based on which is closest
        if abs(self.state[0] - l) < abs(self.state[0] - r):
            start_x = l
            direction_x = 1
        else:
            start_x = r
            direction_x = -1
            
        if abs(self.state[1] - d) < abs(self.state[1] - u):
            start_y = d
            direction_y = 1
        else:
            start_y = u
            direction_y = -1
            
        # Calculate how many passes, floor to be safe and get int
        height = abs(u - d)
        width = abs(r - l)
        nbr_passes = math.ceil(width/max(swath_width, self.turning_radius))
        
        # Reduce nbr passes until distance contraint is satisfied.
        max_distance = max_time * vehicle_speed
        distance = nbr_passes*height + (nbr_passes-1)*self.turning_radius*np.pi/2
        while distance > max_distance:
            swath_width += 2.0
            nbr_passes = math.floor(width/max(swath_width, self.turning_radius))
            distance = nbr_passes*height + (nbr_passes-1)*self.turning_radius*np.pi/2
            
            
        # Calculate changes at each pass. Use Y as long end.
        dx = max(swath_width, 2*self.turning_radius) * direction_x
        dy = abs(height-2*self.turning_radius) * direction_y
        
        # Get stamp
        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()
        h.frame_id = self.map_frame
        lm_path = Path(header=h)
        
        # Calculate starting position
        h.stamp = rospy.Time.now()
        wp1 = PoseStamped(header=h)
        wp1.pose.position.x = start_x + direction_x * swath_width / 2
        wp1.pose.position.y = start_y + direction_x * self.turning_radius
        #lm_path.poses.append(wp1)
        start_x = start_x - direction_x * swath_width / 2
        start_y = start_y + direction_y * self.turning_radius
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