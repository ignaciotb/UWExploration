# Python functionality
from abc import abstractmethod
import pickle
import filelock
import copy

# Math libraries
import dubins
import numpy as np
import torch
import math

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
from MonteCarloTreeClass import MonteCarloTree

class PlannerBase():
    """ Defines basic methods and attributes which are shared amongst
        all planners. Advanced planners inherit from this class.
    """
    def __init__(self, corner_topic, path_topic, planner_req_topic, odom_topic, bounds, turning_radius, training_rate, start_pose,
                 max_time, vehicle_velocity):
        """ Constructor method

        Args:
            corner_topic        (string): publishing topic for corner waypoints
            path_topic          (string): publishing topic for planner waypoints
            planner_req_topic   (string): subscriber topic for callbacks to plan new paths
            odom_topic          (string): subscriber topic for callback to update vehicle odometry
            bounds       (array[double]): [left bound, right bound, upper bound, lower bound]
            turning_radius      (double): the radius on which the vehicle can turn on yaw axis
        """
        self.corner_topic       = corner_topic
        self.path_topic         = path_topic
        self.planner_req_topic  = planner_req_topic
        self.odom_topic         = odom_topic
        self.bounds             = bounds
        self.turning_radius     = turning_radius
        self.training_rate      = training_rate
        self.max_time           = max_time
        self.vehicle_velocity   = vehicle_velocity
        
        # Logic checks
        assert len(self.bounds) == 4, "Wrong number of boundaries given to planner, need specifically 4"
        assert self.turning_radius > 3.0, "Turning radius is way too small"
        
        # Setup class attributes
        self.start_pose = start_pose
        self.state = copy.deepcopy(self.start_pose)
        self.prev_odom  = [0, 0, 0]
        self.gp = SVGP_map(particle_id=0, corners=self.bounds)
        self.distance_travelled = 0
        
        # Corner publisher - needed as boundary for generating inducing points
        self.corner_pub  = rospy.Publisher(self.corner_topic, Path, queue_size=1)
        rospy.sleep(1) # Give time for topic to be registered
        corners = self.generate_ip_corners()
        self.corner_pub.publish(corners)
        
        # Subscribers with callback methods
        rospy.Subscriber(self.planner_req_topic, std_msgs.msg.Bool, self.get_path_cb)
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
        odom_diff = [x-self.prev_odom[0], y-self.prev_odom[1], yaw-self.prev_odom[2]]
        self.prev_odom = [x, y, yaw]
        self.distance_travelled += np.hypot(odom_diff[0], odom_diff[1])
        c = np.cos(-self.start_pose[2])
        s = np.sin(-self.start_pose[2])
        self.state = [self.start_pose[0] + x * c + s * y, self.start_pose[1] - x * s + y * c, self.start_pose[2] + yaw]
        
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
        
        # NOTE: Adding an extra wp closes the pattern, 
        # no need to deform corners then
        corners.poses.append(ul_c)
        
        return corners
    
class SimplePlanner(PlannerBase):
    """ Class for lawnmower pattern planner
    
    Args:
        PlannerBase (obj): Basic template of planner class
    """
    def __init__(self, corner_topic, path_topic, planner_req_topic, odom_topic, 
                 bounds, turning_radius, training_rate, sw, max_time, vehicle_velocity, start_pose):
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
                         bounds, turning_radius, training_rate, start_pose, max_time, vehicle_velocity) 
        
        # Publish path, then train GP
        self.path_pub = rospy.Publisher(path_topic, Path, queue_size=100)
        rospy.sleep(1)
        path = self.generate_path(sw, self.max_time, self.vehicle_velocity)
        self.path_pub.publish(path) 
        self.begin_gp_train()
            
    def get_path_cb(self, msg):
        """ When called, dumps GP

        Args:
            msg (bool): dummy boolean, not used currently
        """
        
        # Freeze a copy of current model for plotting, to let real model keep training
        with self.gp.mutex:
                pickle.dump(self.gp.model, open("GP_env_lawnmower.pickle" , "wb"))
        
        # Notify of current distance travelled
        print("Current distance travelled: " + str(self.distance_travelled) + " m.")
        
    
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
        nbr_passes = math.floor(width/max(swath_width, self.turning_radius))
        
        # Reduce nbr passes until distance contraint is satisfied.
        max_distance = max_time * vehicle_speed
        distance = nbr_passes*height + (nbr_passes-1)*self.turning_radius*np.pi/2
        while distance > max_distance:
            swath_width += 2.0
            nbr_passes = math.floor(width/max(swath_width, self.turning_radius))
            distance = nbr_passes*height + (nbr_passes-1)*self.turning_radius*np.pi/2
            
            
        # Calculate changes at each pass. Use Y as long end.
        dx = max(swath_width, self.turning_radius) * direction_x
        dy = abs(height-2*self.turning_radius) * direction_y
        
        # Get stamp
        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()
        h.frame_id = "map"
        lm_path = Path(header=h)
        
        # Calculate starting position
        h.stamp = rospy.Time.now()
        wp1 = PoseStamped(header=h)
        wp1.pose.position.x = start_x + direction_x * swath_width / 2
        wp1.pose.position.y = start_y 
        lm_path.poses.append(wp1)
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
        
        
class BOPlanner(PlannerBase):
    """ Planner class based on Bayesian Optimization

    Args:
        PlannerBase (obj): Basic template of planner class
    """
    def __init__(self, corner_topic, path_topic, planner_req_topic, odom_topic, bounds, turning_radius, 
                 training_rate, wp_resolution, swath_width, path_nbr_samples, voxel_size, 
                 wp_sample_interval, horizon_distance, border_margin, beta, start_pose,
                 max_time, vehicle_velocity):
        """ Constructor method

        Args:
            corner_topic    (string): publishing topic for corner waypoints
            path_topic      (string): publishing topic for planner waypoints
            bounds   (array[double]): [left bound, right bound, upper bound, lower bound]
            turning_radius  (double): the radius on which the vehicle can turn on yaw axis
        """
        # Invoke constructor of parent class
        super().__init__(corner_topic, path_topic, planner_req_topic, odom_topic, bounds, turning_radius, training_rate, start_pose,
                         max_time, vehicle_velocity) 
        
        # Planner variables
        self.planner_initial_pose = self.state
        self.wp_list              = []
        self.currently_planning   = False
        
        # Path publisher - publishes waypoints for AUV to follow
        self.path_pub    = rospy.Publisher(self.path_topic, Path, queue_size=100)
        rospy.sleep(1)  # Give time for topic to be registered
        
        # Filelocks for file mutexes
        self.gp_env_lock    = filelock.FileLock("GP_env.pickle.lock")
        self.gp_angle_lock  = filelock.FileLock("GP_angle.pickle.lock")
        
        
        # Parameters for optimizer
        self.wp_resolution      = wp_resolution
        self.swath_width        = swath_width
        self.path_nbr_samples   = path_nbr_samples
        self.voxel_size         = voxel_size
        self.wp_sample_interval = wp_sample_interval
        self.horizon_distance   = horizon_distance
        self.border_margin      = border_margin
        self.beta               = beta
        
        # Publish an initial path
        initial_path     = self.initial_sampling_path()
        self.path_pub.publish(initial_path) 
        
        # Setup timers for callbacks
        self.finish_imminent = False
        rospy.Timer(rospy.Duration(2), self.periodic_call)
        self.execute_planner_pub = rospy.Publisher("execute_planning_topic_handle", std_msgs.msg.Bool, queue_size=1)
        self.execute_planner_sub = rospy.Subscriber("execute_planning_topic_handle", std_msgs.msg.Bool, self.execute_planning)
        
        # Initiate training of GP
        self.begin_gp_train(rate=self.training_rate)
        
    def initial_sampling_path(self):
        """ Generates a set of waypoints for initial sampling of BO

        Args:
            n_samples (int): number of samples

        Returns:
            nav_msgs.msg.Path: Waypoint list, in form of poses
        """
        low_x            = max(self.bounds[0] + self.border_margin, min(self.planner_initial_pose[0] - self.horizon_distance, self.planner_initial_pose[0] + self.horizon_distance))
        high_x           = min(self.bounds[1] - self.border_margin, max(self.planner_initial_pose[0] - self.horizon_distance, self.planner_initial_pose[0] + self.horizon_distance))
        low_y            = max(self.bounds[3] + self.border_margin, min(self.planner_initial_pose[1] - self.horizon_distance, self.planner_initial_pose[1] + self.horizon_distance))
        high_y           = min(self.bounds[2] - self.border_margin, max(self.planner_initial_pose[1] - self.horizon_distance, self.planner_initial_pose[1] + self.horizon_distance))
        dynamic_bounds   = [low_x, high_x, high_y, low_y] #self.bounds
        samples = np.random.uniform(low=[dynamic_bounds[0], dynamic_bounds[3], -np.pi], high=[dynamic_bounds[1], dynamic_bounds[2], np.pi], size=[1, 3])
        h = std_msgs.msg.Header()
        h.frame_id = "map"
        h.stamp = rospy.Time.now()
        sampling_path = Path()
        sampling_path.header = h
        for sample in samples:
            path = dubins.shortest_path(self.planner_initial_pose, [sample[0], sample[1], sample[2]], self.turning_radius)
            wp_poses, _ = path.sample_many(self.wp_resolution)
            skip = 1
            if len(wp_poses) == 1:
                skip = 0
            self.wp_list.extend(wp_poses[skip:])
            for pose in wp_poses[skip:]:
                wp = PoseStamped()
                wp.header = h
                wp.pose.position.x = pose[0]
                wp.pose.position.y = pose[1]
                wp.pose.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(0, 0, pose[2]))
                sampling_path.poses.append(wp)
            self.planner_initial_pose = wp_poses[-1]
        return sampling_path
    
    
            
    def get_path_cb(self, msg):
        """ When called, reduces number of wp 

        Args:
            msg (bool): flag for if mission is complete, 0 = Complete
        """
        self.wp_list.pop(0)
            
    def calculate_time2target(self):
        speed = self.vehicle_velocity
        distance = 0
        current = copy.deepcopy(self.state)
        for pose in self.wp_list:
            distance += np.hypot(current[0] - pose[0], current[1] - pose[1])
            current = pose
        return distance/speed
            
    
    def periodic_call(self, msg):
        time2target = self.calculate_time2target()
        print("Time to target: " + str(time2target))
        if time2target < 45 and self.currently_planning == False:
            self.currently_planning = True
            self.execute_planner_pub.publish(True)
            print("Beginning planning...")
        
        if time2target < 15 and self.finish_imminent == False:
            self.finish_imminent = True
            print("Stopping planning...")
            
            
    def execute_planning(self, msg):
        
        # Freeze a copy of current model for planning, to let real model keep training
        with self.gp.mutex:
                pickle.dump(self.gp.model, open("GP_env.pickle" , "wb"))
                
        with self.gp_env_lock:
            model = pickle.load(open("GP_env.pickle","rb"))
            
        # Get a new candidate trajectory with Bayesian optimization
        low_x            = max(self.bounds[0] + self.border_margin, min(self.planner_initial_pose[0] - self.horizon_distance, self.planner_initial_pose[0] + self.horizon_distance))
        high_x           = min(self.bounds[1] - self.border_margin, max(self.planner_initial_pose[0] - self.horizon_distance, self.planner_initial_pose[0] + self.horizon_distance))
        low_y            = max(self.bounds[3] + self.border_margin, min(self.planner_initial_pose[1] - self.horizon_distance, self.planner_initial_pose[1] + self.horizon_distance))
        high_y           = min(self.bounds[2] - self.border_margin, max(self.planner_initial_pose[1] - self.horizon_distance, self.planner_initial_pose[1] + self.horizon_distance))
        dynamic_bounds   = [low_x, high_x, high_y, low_y] #self.bounds
        
        # Signature in: Gaussian Process of terrain, xy bounds where we can find solution, current pose
        
        MCTS = MonteCarloTree(self.state[:2], model, beta=self.beta, bounds=self.bounds,
                              horizon_distance=self.horizon_distance, border_margin=self.border_margin, 
                              max_depth=3)
        # NOTE: working on setting this to iterate until we are out of time. Then it has to yield
        # a best solution (just implemented get best)
        # also have to change so that the returned candidate is a tensor, and we have enough time 
        # to run theta optimization
        
        BO                          = BayesianOptimizer(gp_terrain=model, bounds=dynamic_bounds, beta=self.beta, 
                                                        current_pose=self.planner_initial_pose, wp_resolution=self.wp_resolution, 
                                                        turning_radius=self.turning_radius, swath_width=self.swath_width, 
                                                        path_nbr_samples=self.path_nbr_samples, voxel_size=self.voxel_size, 
                                                        wp_sample_interval=self.wp_sample_interval)
        
        candidates_XY1               = BO.optimize_XY_with_grad(max_iter=5, nbr_samples=200)
        
        while self.finish_imminent == False:
            MCTS.iterate()
        candidates_XY = MCTS.get_best_solution().position
        
        candidates_XY = (torch.from_numpy(np.array([candidates_XY]))).type(torch.FloatTensor)
        
        print(candidates_XY1)
        print(candidates_XY)
        
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
        path = dubins.shortest_path(self.planner_initial_pose, [location[0], location[1], location[2]], self.turning_radius)
        wp_poses, _ = path.sample_many(self.wp_resolution)
        skip = 1
        if len(wp_poses) == 1:
            skip = 0
        self.wp_list.extend(wp_poses[skip:])
        for pose in wp_poses[skip:]:       #removing first as hack to ensure AUV doesnt get stuck
            wp = PoseStamped()
            wp.header = h
            wp.pose.position.x = pose[0] 
            wp.pose.position.y = pose[1]  
            wp.pose.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(0, 0, pose[2]))
            sampling_path.poses.append(wp)
        self.planner_initial_pose = wp_poses[-1]
        self.path_pub.publish(sampling_path)
        self.currently_planning = False
        self.finish_imminent = False
        print("Current distance travelled: " + str(self.distance_travelled) + " m.")
        