# Torch libraries
import torch
import botorch

# Math libraries
import dubins
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
import MonteCarloTreeClass
import BayesianOptimizerClass

class BOPlanner(PlannerTemplateClass.PlannerTemplate):
    """ Planner which uses Bayesian Optimization and MCTS

    Args:
        PlannerTemplate (obj): Basic template of planner class
    """
    def __init__(self, corner_topic, path_topic, planner_req_topic, odom_topic, bounds, turning_radius, 
                 training_rate, wp_resolution, swath_width, path_nbr_samples, voxel_size, 
                 wp_sample_interval, horizon_distance, border_margin, beta,
                 max_time, vehicle_velocity, MCTS_begin_time, MCTS_interrupt_time):
        
        """ Constructor method

        Args:
            corner_topic        (string): publishing topic for corner waypoints
            path_topic          (string): publishing topic for planner waypoints
            planner_req_topic   (string): _description_
            odom_topic          (string): _description_
            bounds        (list[double]): [left bound, right bound, upper bound, lower bound]
            turning_radius      (double): the radius on which the vehicle can turn on yaw axis
            training_rate          (int): _description_
            wp_resolution       (_type_): _description_
            swath_width         (_type_): _description_
            path_nbr_samples    (_type_): _description_
            voxel_size          (_type_): _description_
            wp_sample_interval  (_type_): _description_
            horizon_distance    (_type_): _description_
            border_margin       (_type_): _description_
            beta                (_type_): _description_
            max_time            (_type_): _description_
            vehicle_velocity    (_type_): _description_
            MCTS_begin_time     (_type_): _description_
            MCTS_interrupt_time (_type_): _description_
        """
        # Invoke constructor of parent class
        super().__init__(corner_topic, path_topic, planner_req_topic, odom_topic, bounds, turning_radius, training_rate,
                         max_time, vehicle_velocity) 
        
        # Planner variables
        self.planner_initial_pose   = copy.deepcopy(self.state)
        self.wp_list                = []
        self.currently_planning     = False
        self.nbr_wp                 = 0

        # Path publisher - publishes waypoints for AUV to follow
        self.path_pub               = rospy.Publisher(self.path_topic, Path, queue_size=100)
        rospy.sleep(1)  # Give time for topic to be registered
        
        # Filelocks for file mutexes
        self.gp_env_lock            = filelock.FileLock("GP_env.pickle.lock")
        self.gp_angle_lock          = filelock.FileLock("GP_angle.pickle.lock")
        
        # Parameters for optimizer
        self.wp_resolution          = wp_resolution
        self.swath_width            = swath_width
        self.path_nbr_samples       = path_nbr_samples
        self.voxel_size             = voxel_size
        self.wp_sample_interval     = wp_sample_interval
        self.horizon_distance       = horizon_distance
        self.border_margin          = border_margin
        self.beta                   = beta
        self.MCTS_begin_time        = MCTS_begin_time
        self.MCTS_interrupt_time    = MCTS_interrupt_time
        
        # Publish an initial path
        initial_path     = self.initial_deterministic_path()
        self.path_pub.publish(initial_path) 
        
        # Setup timers for callbacks
        self.finish_imminent = False
        rospy.Timer(rospy.Duration(2), self.periodic_call)
        self.execute_planner_pub = rospy.Publisher("execute_planning_topic_handle", std_msgs.msg.Bool, queue_size=1)
        self.execute_planner_sub = rospy.Subscriber("execute_planning_topic_handle", std_msgs.msg.Bool, self.execute_planning)
        
        # Initiate training of GP
        self.begin_gp_train(rate=self.training_rate)
    
    def initial_deterministic_path(self):
        """ Generates a path to the center of the map, as defined by the boundaries.

        Returns:
            nav_msgs.msg.Path: ROS message with waypoint poses in list
        """
        
        while not self.odom_init and not rospy.is_shutdown():
            print("Planner is waiting for odometry before starting.")
            rospy.sleep(2)

        # Generate point in center of bounds
        x_pos = self.bounds[0] + (self.bounds[2] - self.bounds[0])/2
        y_pos = self.bounds[1] + (self.bounds[3] - self.bounds[1])/2
        samples = np.random.uniform(low=[x_pos - 1, y_pos - 1, -np.pi], high=[x_pos + 1, y_pos + 1, np.pi], size=[1, 3])
        h = std_msgs.msg.Header()
        h.frame_id = self.map_frame
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
                self.nbr_wp += 1
            self.planner_initial_pose = wp_poses[-1]
        return sampling_path
        
    def initial_random_path(self, nbr_samples = 1):
        """ Generates a set of random waypoints for initial sampling of BO

        Args:
            nbr_samples (int): number of samples. Defaults to 1.

        Returns:
            nav_msgs.msg.Path: ROS message with waypoint poses in list
        """
        low_x            = max(self.bounds[0] + self.border_margin, min(self.planner_initial_pose[0] - self.horizon_distance, self.planner_initial_pose[0] + self.horizon_distance))
        low_y            = max(self.bounds[1] + self.border_margin, min(self.planner_initial_pose[1] - self.horizon_distance, self.planner_initial_pose[1] + self.horizon_distance))
        high_x           = min(self.bounds[2] - self.border_margin, max(self.planner_initial_pose[0] - self.horizon_distance, self.planner_initial_pose[0] + self.horizon_distance))
        high_y           = min(self.bounds[3] - self.border_margin, max(self.planner_initial_pose[1] - self.horizon_distance, self.planner_initial_pose[1] + self.horizon_distance))
        dynamic_bounds   = [low_x, high_x, high_y, low_y] #self.bounds
        samples = np.random.uniform(low=[dynamic_bounds[0], dynamic_bounds[3], -np.pi], high=[dynamic_bounds[1], dynamic_bounds[2], np.pi], size=[nbr_samples, 3])
        h = std_msgs.msg.Header()
        h.frame_id = self.map_frame
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
                self.nbr_wp += 1
            self.planner_initial_pose = wp_poses[-1]
        return sampling_path
    
    def update_wp_cb(self, msg):
        """ When called, reduces number of waypoints tracked.

        Args:
            msg (bool): not currently used
        """
        self.wp_list.pop(0)
        self.nbr_wp -= 1
            
    def calculate_time2target(self):
        """ Calculates the time to target based on euclidean distance
            along the path.

        Returns:
            double: time to arrival at target location
        """
        speed = self.vehicle_velocity
        distance = 0
        current = copy.deepcopy(self.state)
        for pose in self.wp_list:
            distance += np.hypot(current[0] - pose[0], current[1] - pose[1])
            current = pose
        return distance/speed
            
    
    def periodic_call(self, msg):
        """ Timer based callback which initiates planner actions.

        Args:
            msg (bool): not used.
        """
        #time2target = self.calculate_time2target()
        if self.nbr_wp < 2 and self.currently_planning == False:
            self.currently_planning = True
            self.execute_planner_pub.publish(True)
            print("Beginning planning...")
        
        if self.nbr_wp < 1 and self.finish_imminent == False:
            self.finish_imminent = True
            print("Stopping planning...")
            
    def execute_planning(self, msg):
        """_summary_

        Args:
            msg (_type_): _description_
        """
        
        # Freeze a copy of current model for planning, to let real model keep training
        with self.gp.mutex:
                #pickle.dump(self.gp.model, open("GP_env.pickle" , "wb"))
                torch.save({'model' : self.gp.model.state_dict()}, "GP_env.pickle")
                
        with self.gp_env_lock:
            cp = torch.load("GP_env.pickle")
            self.frozen_gp.load_state_dict(cp['model'])
            
        # Get a new candidate trajectory with Bayesian optimization
        low_x            = max(self.bounds[0] + self.border_margin, min(self.planner_initial_pose[0] - self.horizon_distance, self.planner_initial_pose[0] + self.horizon_distance))
        low_y            = max(self.bounds[3] + self.border_margin, min(self.planner_initial_pose[1] - self.horizon_distance, self.planner_initial_pose[1] + self.horizon_distance))
        high_x           = min(self.bounds[1] - self.border_margin, max(self.planner_initial_pose[0] - self.horizon_distance, self.planner_initial_pose[0] + self.horizon_distance))
        high_y           = min(self.bounds[2] - self.border_margin, max(self.planner_initial_pose[1] - self.horizon_distance, self.planner_initial_pose[1] + self.horizon_distance))
        dynamic_bounds   = [low_x, low_y, high_x, high_y] #self.bounds
        
        # Signature in: Gaussian Process of terrain, xy bounds where we can find solution, current pose
        MCTS = MonteCarloTreeClass.MonteCarloTree(self.state[:2], self.frozen_gp, beta=self.beta, bounds=self.bounds,
                              horizon_distance=self.horizon_distance, border_margin=self.border_margin)
        
        while self.finish_imminent == False:
            MCTS.iterate()
        
        rush_order_activated = False
        
        try:
            candidates_XY = MCTS.get_best_solution().position
            candidates_XY = (torch.from_numpy(np.array([candidates_XY]))).type(torch.FloatTensor)
        
        except:
            # If MCTS fails (due to a rush order) then give a myopic quick candidate
            rospy.loginfo("MCTS failed to get candidate, fallback used.")
            rush_order_activated  = True
            self.bounds_XY_torch  = torch.tensor([[self.bounds[0], self.bounds[1]], [self.bounds[2], self.bounds[3]]]).to(torch.float)
            self.XY_acqf          = botorch.acquisition.UpperConfidenceBound(model=self.frozen_gp, beta=self.beta)
            candidates_XY, _      = botorch.optim.optimize_acqf(acq_function=self.XY_acqf, bounds=self.bounds_XY_torch, q=1, num_restarts=5, raw_samples=50)
        
        print("XY cand ", candidates_XY)
        
        print("Optimizing angle...")
        BO = BayesianOptimizerClass.BayesianOptimizer(self.state, self.frozen_gp, wp_resolution=self.wp_resolution,
                                                      turning_radius=self.turning_radius, swath_width=self.swath_width,
                                                      path_nbr_samples=self.path_nbr_samples, voxel_size=self.voxel_size,
                                                      wp_sample_interval=self.wp_sample_interval)
        
        angle_optim_max_iter = 5
        if rush_order_activated:
            print("Rush order requested!")
            angle_optim_max_iter = 0

        candidates_theta, angle_gp  = BO.optimize_theta_with_grad(XY=candidates_XY, max_iter=angle_optim_max_iter, nbr_samples=15)
        
        candidate                   = torch.cat([candidates_XY, candidates_theta], 1).squeeze(0)
        
        print("Optimal pose: ", candidate, ", publishing trajectory.")
        
        # Publish this trajectory as a set of waypoints
        h = std_msgs.msg.Header()
        h.stamp = rospy.Time.now()
        h.frame_id = self.map_frame
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
            self.nbr_wp += 1
        self.planner_initial_pose = wp_poses[-1]
        self.path_pub.publish(sampling_path)
        self.currently_planning = False
        self.finish_imminent = False
        with self.gp_angle_lock:
            print("Saving stuff--------------------")
            torch.save({"model": angle_gp.state_dict()}, self.store_path  + "_GP_" + str(round(self.distance_travelled)) + "_angle.pickle")
            torch.save({"model": self.frozen_gp.state_dict()}, self.store_path + "_GP_" + str(round(self.distance_travelled)) + "_env.pickle")
        print("Current distance travelled: " + str(round(self.distance_travelled)) + " m.")
        