# Python functionality
from abc import abstractmethod
import pickle
import filelock

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
import tf
import tf_conversions
import geometry_msgs

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
        assert len(self.bounds) == 4, "Wrong number of boundaries given to planner, need specifically 4"
        assert self.turning_radius > 3.0, "Turning radius is way too small"
        
        # Setup class attributes
        self.state = [0, 0, 0]
        self.gp = SVGP_map(particle_id=0, corners=self.bounds)
        self.distance_travelled = 0
        
        # Corner publisher - needed as boundary for generating inducing points
        self.corner_pub  = rospy.Publisher(self.corner_topic, Path, queue_size=1)
        rospy.sleep(1) # Give time for topic to be registered
        corners = self.generate_ip_corners()
        self.corner_pub.publish(corners)
        
        # Subscribers with callback methods
        rospy.Subscriber("/navigation/hugin_0/wp_status", std_msgs.msg.Bool, self.get_path_cb)
        rospy.Subscriber("/sim/hugin_0/odom", Odometry, self.odom_state_cb)
        
    @abstractmethod
    def get_path_cb():
        """ Abstract callback method, called when no more waypoints left.
        """
        pass
    
    def begin_gp_train(self, rate = 10):
        """ Begin training of GP, gives up control of thread

        Args:
            rate (int, optional): Training rate in Hz. Defaults to 10.
        """
        r = rospy.Rate(rate)
        while not rospy.is_shutdown():
            self.gp.train_iteration()  
            r.sleep()
            
    def odom_state_cb(self, msg):
        """ Gets our current 2D state (x,y,theta) from the odometry topic.
            Also calculates the total distance travelled.

        Args:
            msg (_type_): A pose message
        """
        explicit_quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(explicit_quat)
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.distance_travelled += np.hypot(x-self.state[0], y-self.state[1])
        self.state = [x, y, yaw]
        
    def generate_ip_corners(self):
        """ Generates corners of the bounding area for inducing point generation

        Returns:
            corners (nav_msgs.msg.Path): four waypoints bounding the area in a rectangle
        """
        corners = Path(header=std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id="map"))
        
        l = self.bounds[0]
        r = self.bounds[1]
        u = self.bounds[2]
        d = self.bounds[3]
        
        # Append corners
        ul_c = PoseStamped()
        ul_c.pose.position.x = l 
        ul_c.pose.position.y = u
        corners.poses.append(ul_c)
        
        ur_c = PoseStamped()
        ur_c.pose.position.x = r
        ur_c.pose.position.y = u 
        corners.poses.append(ur_c)

        # NOTE: switched order of dr_c and dl_c being appended to make
        #       visualization of borders easier in RVIZ

        dr_c = PoseStamped()
        dr_c.pose.position.x = r
        dr_c.pose.position.y = d
        corners.poses.append(dr_c)
        
        dl_c = PoseStamped()
        dl_c.pose.position.x = l
        dl_c.pose.position.y = d
        corners.poses.append(dl_c)
        
        # Adding an extra wp closes the pattern, 
        # no need to deform corners then
        corners.poses.append(ul_c)
        
        return corners
    
class SimplePlanner(PlannerBase):
    """ Class for lawnmower pattern planner
    
    Args:
        PlannerBase (obj): Basic template of planner class
    """
    def __init__(self, corner_topic, path_topic, bounds, turning_radius, sw, so):
        # Invoke constructor of parent class
        super().__init__(corner_topic, path_topic, bounds, turning_radius) 
        self.path_pub    = rospy.Publisher(path_topic, Path, queue_size=100)
        rospy.sleep(1)
        path = self.generate_path(sw, so)
        self.path_pub.publish(path) 
        self.begin_gp_train()
            
    def get_path_cb(self, msg):
        """ When called, dumps GP

        Args:
            msg (bool): dummy boolean, not used currently
        """
        
        # Freeze a copy of current model for plotting, to let real model keep training
        with self.gp.mutex:
                pickle.dump(self.gp.model, open("GP_env.pickle" , "wb"))
        
        # Notify of current distance travelled
        print("Current distance travelled: " + str(self.distance_travelled) + " m.")
        
    
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
        # Invoke constructor of parent class
        super().__init__(corner_topic, path_topic, bounds, turning_radius) 
        
        # Path publisher - publishes waypoints for AUV to follow
        self.path_pub    = rospy.Publisher(self.path_topic, Path, queue_size=100)
        rospy.sleep(1)  # Give time for topic to be registered
        
        # Publish an initial path
        initial_path     = self.initial_sampling_path(n_samples=1)
        self.path_pub.publish(initial_path) 
        
        # Filelocks for file mutexes
        self.gp_env_lock    = filelock.FileLock("GP_env.pickle.lock")
        self.gp_angle_lock  = filelock.FileLock("GP_angle.pickle.lock")
        
        # Initiate training of GP
        # TODO: why set the training rate at 30 hz? Instead set it to 10 Hz
        # On laptop, training takes ~0.05 seconds, which means 30 hz is too fast.
        # Don't know yet about jetson.
        self.begin_gp_train()
        
    def initial_sampling_path(self, n_samples):
        """ Generates a set of waypoints for initial sampling of BO

        Args:
            n_samples (int): number of samples

        Returns:
            nav_msgs.msg.Path: Waypoint list, in form of poses
        """
        samples = np.random.uniform(low=[self.bounds[0], self.bounds[3]], high=[self.bounds[1], self.bounds[2]], size=[n_samples, 2])
        h = std_msgs.msg.Header()
        h.frame_id = "map"
        h.stamp = rospy.Time.now()
        sampling_path = Path()
        sampling_path.header = h
        for sample in samples:
            wp = PoseStamped()
            wp.header = h
            wp.pose.position.x = -40 #sample[0] #-20 
            wp.pose.position.y = 15 #sample[1] #10 
            sampling_path.poses.append(wp)
        return sampling_path
            
    def get_path_cb(self, msg):
        """ When called, calls optimization to find the best new path to take.

        Args:
            msg (_type_): _description_
        """
        
        # Freeze a copy of current model for planning, to let real model keep training
        with self.gp.mutex:
                pickle.dump(self.gp.model, open("GP_env.pickle" , "wb"))
                
        with self.gp_env_lock:
            model = pickle.load(open("GP_env.pickle","rb"))
            
        # Get a new candidate trajectory with Bayesian optimization
        horizon_distance = 60
        border_margin    = 10
        low_x            = max(self.bounds[0] + border_margin, min(self.state[0] - horizon_distance, self.state[0] + horizon_distance))
        high_x           = min(self.bounds[1] - border_margin, max(self.state[0] - horizon_distance, self.state[0] + horizon_distance))
        low_y            = max(self.bounds[3] + border_margin, min(self.state[1] - horizon_distance, self.state[1] + horizon_distance))
        high_y           = min(self.bounds[2] - border_margin, max(self.state[1] - horizon_distance, self.state[1] + horizon_distance))
        dynamic_bounds   = [low_x, high_x, high_y, low_y] #self.bounds
        
        # Signature in: Gaussian Process of terrain, xy bounds where we can find solution, current pose
        BO                          = BayesianOptimizer(gp_terrain=model, bounds=dynamic_bounds, beta=30.0, current_pose=self.state)
        candidates_XY               = BO.optimize_XY_with_grad(max_iter=5, nbr_samples=200)
        candidates_theta, angle_gp  = BO.optimize_theta_with_grad(XY=candidates_XY, max_iter=5, nbr_samples=15)
        candidate                   = torch.cat([candidates_XY, candidates_theta], 1).squeeze(0)
        
        with self.gp_angle_lock:
                pickle.dump(angle_gp, open("GP_angle.pickle" , "wb"))
        
        # Signature out: Candidate (xy theta)
        print(candidate)
        #print(value)
        
        # Publish this trajectory as a set of waypoints
        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()
        h.frame_id = "map"
        sampling_path = Path()
        sampling_path.header = h
        location = candidate.numpy()
        path = dubins.shortest_path(self.state, [location[0], location[1], location[2]], self.turning_radius)
        wp_poses, _ = path.sample_many(5)
        for pose in wp_poses[2:]:       #removing first as hack to ensure AUV doesnt get stuck
            wp = PoseStamped()
            wp.header = h
            wp.pose.position.x = pose[0] 
            wp.pose.position.y = pose[1]  
            wp.pose.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(0, 0, pose[2]))
            sampling_path.poses.append(wp)
        if len(sampling_path.poses) == 0:
            wp = PoseStamped()
            wp.header = h
            wp.pose.position.x = self.state[0] + 5*np.cos(self.state[2]) 
            wp.pose.position.y = self.state[1] + 5*np.cos(self.state[2]) 
            wp.pose.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(0, 0, self.state[2]))
            sampling_path.poses.append(wp)
            
        self.path_pub.publish(sampling_path)
        print("Current distance travelled: " + str(self.distance_travelled) + " m.")
        