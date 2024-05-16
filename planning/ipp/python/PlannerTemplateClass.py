# Python functionality
from abc import abstractmethod
import time

# Math libraries
import numpy as np

# ROS imports
import rospy
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
import std_msgs.msg
import tf.transformations
import tf

# Custom libraries
import GaussianProcessClass

class PlannerTemplate(object):
    """ Defines basic methods and attributes which are shared amongst
        all planners. Advanced planners inherit from this class.
    """
    def __init__(self, corner_topic, path_topic, planner_req_topic, odom_topic, bounds, 
                 turning_radius, training_rate, max_time, vehicle_velocity):
        
        """ Constructor method

        Args:
            corner_topic        (string): publishing topic for corner waypoints
            path_topic          (string): publishing topic for planner waypoints
            planner_req_topic   (string): subscriber topic for callbacks to plan new paths
            odom_topic          (string): subscriber topic for callback to update vehicle odometry
            bounds        (list[double]): [low_x, low_y, high_x, high_y]
            turning_radius      (double): the radius on which the vehicle can turn on yaw axis
            training_rate          (int): rate at which GP is trained
            max_time            (double): expected maximum mission time
            vehicle_velocity    (double): expected average vehicle velocity
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
        
        # Frame transforms
        self.map_frame          = rospy.get_param("~map_frame")
        self.odom_frame         = rospy.get_param("~odom_frame")
        self.tf_listener        = tf.TransformListener()
        
        # Storage handle
        self.store_path         = "Results_" + time.ctime()
        
        # Logic checks
        assert len(self._bounds) == 4, "Planner boundary dimensions wrong, need specifically 4"
        assert self.odom_topic != "", "Planner topic empty"
        assert self.path_topic != "", "Planner topic empty"
        assert self.corner_topic != "", "Planner topic empty"
        assert self.turning_radius > 2.0, "Planner turning radius too small"
        assert self.odom_topic is int, "Planner training rate not integer value"
        
        # Setup class attributes
        self.state              = []
        self.gp                 = GaussianProcessClass.SVGP_map(particle_id=0, corners=self.bounds)
        self.distance_travelled = 0
        
        # Corner publisher - needed as boundary for generating inducing points
        self.corner_pub  = rospy.Publisher(self.corner_topic, Path, queue_size=1)
        rospy.sleep(1) # Give time for topic to be registered
        corners = self.generate_ip_corners()
        self.corner_pub.publish(corners)
        
        # Subscribers with callback methods
        self.odom_init = False
        rospy.Subscriber(self.planner_req_topic, std_msgs.msg.Bool, self.update_wp_cb)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_update_cb)
        rospy.sleep(1)
        
    @abstractmethod
    def update_wp_cb():
        """ Abstract callback method, called when no more waypoints left.
        """
        pass
    
    def begin_gp_train(self, rate = 30):
        """ Begin handling of GP batch training in this thread

        Args:
            rate (int, optional): Training rate in Hz. Defaults to 30.
        """
        r = rospy.Rate(rate)
        while not rospy.is_shutdown():
            self.gp.train_iteration()  
            r.sleep()
            
    def odom_update_cb(self, msg):
        """ Gets our current 2D state (x,y,theta) from the tf frames.
            Also calculates the total distance travelled.

        Args:
            msg (PoseWithCovarianceStamped): A pose message (assumes msg type from AUV messages)
        """

        p = PoseStamped(header=msg.header, pose=msg.pose.pose)
        p.header.stamp = msg.header.stamp
        self.tf_listener.waitForTransform(self.map_frame, p.header.frame_id, rospy.rostime.Duration(0, 1e8)) # 0.1s
        p_in_map = self.tf_listener.transformPose(self.map_frame, p)
        explicit_quat = [p_in_map.pose.orientation.x, p_in_map.pose.orientation.y, p_in_map.pose.orientation.z, p_in_map.pose.orientation.w]
        _, _, yaw = tf.transformations.euler_from_quaternion(explicit_quat)
        if len(self.state) > 0:
            self.distance_travelled += np.hypot(p_in_map.pose.position.x - self.state[0], p_in_map.pose.position.y - self.state[1])
        self.state = [p_in_map.pose.position.x, p_in_map.pose.position.y, yaw]
        self.odom_init = True
        
    def generate_ip_corners(self):
        """ Generates rectangle corners of the bounding area for inducing point generation

        Returns:
            corners (nav_msgs.msg.Path): four waypoints bounding the area in a rectangle
        """
        corners = Path(header=std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id=self.map_frame))
        
        low_x   = self.bounds[0]
        low_y   = self.bounds[1]
        high_x  = self.bounds[2]
        high_y  = self.bounds[3]
        
        # Append corners
        ul_c = PoseStamped(header=std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id=self.map_frame))
        ul_c.pose.position.x = low_x
        ul_c.pose.position.y = high_y
        corners.poses.append(ul_c)
        
        ur_c = PoseStamped(header=std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id=self.map_frame))
        ur_c.pose.position.x = high_x
        ur_c.pose.position.y = high_y 
        corners.poses.append(ur_c)

        # NOTE: switched order of dr_c and dl_c being appended to make
        #       visualization of borders easier in RVIZ

        dr_c = PoseStamped(header=std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id=self.map_frame))
        dr_c.pose.position.x = high_x
        dr_c.pose.position.y = low_y
        corners.poses.append(dr_c)
        
        dl_c = PoseStamped(header=std_msgs.msg.Header(stamp=rospy.Time.now(), frame_id=self.map_frame))
        dl_c.pose.position.x = low_x
        dl_c.pose.position.y = low_y
        corners.poses.append(dl_c)
        
        # NOTE: Adding an extra wp closes the pattern, 
        # no need to deform corners then
        corners.poses.append(ul_c)
        
        return corners