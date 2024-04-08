# Python functionality
from abc import abstractmethod
import copy
import pickle

# Math libraries
import dubins
import numpy as np
import torch

# ROS imports
import rospy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
import std_msgs.msg
import tf.transformations

# BoTorch
from botorch.models.approximate_gp import SingleTaskVariationalGP

# Custom libraries
from BayesianOptimizerClass import BayesianOptimizer
from GaussianProcessClass import SVGP_map

class PlannerBase():
    """ Defines basic methods and attributes which are shared amongst
        all planners. Advanced planners inherit from this class.
    """
    def __init__(self, corner_topic, path_topic, bounds, turning_radius):
        """ Constructor method

        Args:
            corner_topic    (string): publishing topic for corner waypoints
            path_topic      (string): publishing topic for planner waypoints
            bounds   (array[double]): [left bound, right bound, upper bound, lower bound]
            turning_radius  (double): the radius on which the vehicle can turn on yaw axis
        """
        self.corner_topic   = corner_topic
        self.path_topic     = path_topic
        self.bounds         = bounds
        self.turning_radius = turning_radius
        
        # Logic checks
        assert len(self.bounds) == 4, "Wrong number of boundaries given"
        assert self.turning_radius > 0.1, "Turning radius is too small"
        
    def generate_ip_corners(self):
        """ Generates corners of the bounding area for inducing point generation

        Returns:
            corners (nav_msgs.msg.Path): four waypoints bounding the area 
        """
        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()
        corners = Path()
        corners.header = h
        
        l = self.bounds[0]
        r = self.bounds[1]
        u = self.bounds[2]
        d = self.bounds[3]
        
        # Note: Deform rectangle slightly. This helps generating mesh with 
        # open3d for inducing point generation.

        # Append corners
        ul_c = PoseStamped()
        ul_c.pose.position.x = l + 3
        ul_c.pose.position.y = u - 4
        ul_c.header = h
        corners.poses.append(ul_c)
        
        ur_c = PoseStamped()
        ur_c.header = h
        ur_c.pose.position.x = r -3
        ur_c.pose.position.y = u + 7
        corners.poses.append(ur_c)

        dl_c = PoseStamped()
        dl_c.header = h
        dl_c.pose.position.x = l + 4
        dl_c.pose.position.y = d - 4
        corners.poses.append(dl_c)

        dr_c = PoseStamped()
        dr_c.header = h
        dr_c.pose.position.x = r
        dr_c.pose.position.y = d
        corners.poses.append(dr_c)
        
        return corners
    
class SimplePlanner(PlannerBase):
    """ Class for lawnmower pattern planner
    
    Args:
        PlannerBase (obj): Basic template of planner class
    """
    
    def generate_path(self, sW, sO, fM = False):
        """ Creates a basic path in lawnmower pattern in a rectangle with given parameters

        Args:
            sW          (double): Swath width. 
            sO          (double): Swath overlap. 0.0 <= fO < 1.0
            fM  (bool, optional): Strict full coverage of entire rectangle. Defaults to False.

        Returns:
            nav_msgs.msg.Path: Waypoint list, in form of poses
        """
        
        l = self.bounds[0]
        r = self.bounds[1]
        u = self.bounds[2]
        d = self.bounds[3]
        
        tR = self.turning_radius
        
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
        lm_path = Path()
        lm_path.header = h
        
        
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

        return lm_path 
        
        
class BOPlanner(PlannerBase):
    """ Planner class based on Bayesian Optimization

    Args:
        PlannerBase (obj): Basic template of planner class
    """
    def __init__(self, corner_topic, path_topic, bounds, turning_radius):
        """ Constructor method

        Args:
            corner_topic    (string): publishing topic for corner waypoints
            path_topic      (string): publishing topic for planner waypoints
            bounds   (array[double]): [left bound, right bound, upper bound, lower bound]
            turning_radius  (double): the radius on which the vehicle can turn on yaw axis
        """
        super().__init__(corner_topic, path_topic, bounds, turning_radius) 
        self.state = [0, 0, 0]
        self.gp = SVGP_map(0, "botorch", self.bounds)
        self.corner_pub  = rospy.Publisher(self.corner_topic, Path, queue_size=1)
        self.path_pub    = rospy.Publisher(self.path_topic, Path, queue_size=10)
        corners = self.generate_ip_corners()
        self.corner_pub.publish(corners)
        initial_path = self.initial_sampling_path(n_samples=1)
        self.path_pub.publish(initial_path) 
        rospy.Subscriber("/navigation/hugin_0/wp_status", std_msgs.msg.Bool, self.get_path_cb)
        rospy.Subscriber("/sim/hugin_0/odom", Odometry, self.odom_state_cb)
        
        
        # TODO: why set the training rate at 30 hz?
        
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.gp.train_iteration()  
            r.sleep()
        
        #rospy.spin()
    
    def cost_function(self, current_pose, suggested_pose, wp_resolution = 0.5):
        """ Calculates cost in terms of dubins path between poses

        Args:
            current_pose             (_type_): _description_
            suggested_pose           (_type_): _description_
            wp_resolution   (float, optional): _description_. Defaults to 0.5.

        Returns:
            double: cost of path
        """

        path = dubins.shortest_path(current_pose, suggested_pose, self.turning_radius)
        _, length_arr = path.sample_many(wp_resolution) # _ = waypoints, if needed for future reference
        cost = length_arr[-1] + wp_resolution
        return cost
        
    def initial_sampling_path(self, n_samples):
        """ Generates a set of waypoints for initial sampling of BO

        Args:
            n_samples (int): number of samples

        Returns:
            nav_msgs.msg.Path: Waypoint list, in form of poses
        """
        samples = np.random.uniform(low=[self.bounds[0], self.bounds[3]], high=[self.bounds[1], self.bounds[2]], size=[n_samples, 2])
        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()
        sampling_path = Path()
        sampling_path.header = h
        for sample in samples:
            wp = PoseStamped()
            wp.header = h
            wp.pose.position.x = sample[0]
            wp.pose.position.y = sample[1]
            sampling_path.poses.append(wp)
        return sampling_path
            
    def get_path_cb(self, msg):
        #PATH = "model.pt"
        #print(self.gp.model.model)
        #torch.save(self.gp.model.state_dict(), PATH)
        #device = torch.device('cpu')    
        #initial_x = self.gp.model._original_train_inputs()
        #model = SingleTaskVariationalGP(initial_x)
        #model.load_state_dict(torch.load(PATH, map_location=device))
        
        # TODO: fix "dictionary changed size during iteration", caused by 
        # pickling object changing size while it is being dumped.
        # Fundamentally the issue with deepcopy as well, cannot
        # deepcopy an object with grad enabled.
        # Current solution is a hack to keep attempting pickle.
        
        pickled = False
        while pickled == False:
            try:
                with open("" + "model_name.dump" , "wb") as f:
                    pickle.dump(self.gp.model, f)
                
                model = pickle.load(open("" + "model_name.dump","rb"))
                pickled = True
            except:
                pass

        
        current_pose = []
        BO = BayesianOptimizer(gp=model, bounds=self.bounds, beta=2.0, current_pose=self.state)
        candidate, value = BO.optimize()
        print(candidate)
        print(value)
        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()
        sampling_path = Path()
        sampling_path.header = h
        wp = PoseStamped()
        location = candidate.numpy()
        wp.pose.position.x = location[0][0]
        wp.pose.position.y = location[0][1]
        wp.header = h
        sampling_path.poses.append(wp)
        self.path_pub.publish(sampling_path)


    def odom_state_cb(self, msg):
        explicit_quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(explicit_quat)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.state = [x, y, yaw]
        