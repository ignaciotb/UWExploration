#!/usr/bin/env python3

# Standard tools
import numpy as np
import dubins
import copy

# ROS imports
import rospy
import ipp_utils
import tf
import tf_conversions
import geometry_msgs
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import PoseStamped
import std_msgs.msg
import tf.transformations
from sensor_msgs.msg import PointCloud2, PointField
import actionlib
from ipp.msg import PathPlanAction, PathPlanResult
import tf2_ros
from tf.transformations import translation_matrix, quaternion_matrix 
from std_msgs.msg import Bool
from barfoot_utils_np import *


def matrix_from_tf(transform):
    if transform._type == 'geometry_msgs/TransformStamped':
        transform = transform.transform

    trans = (transform.translation.x,
             transform.translation.y,
             transform.translation.z)
    quat_ = (transform.rotation.x,
             transform.rotation.y,
             transform.rotation.z,
             transform.rotation.w)

    tmat = translation_matrix(trans)
    qmat = quaternion_matrix(quat_)

    return np.dot(tmat, qmat)


class MaxIPPPlaner():
    
    def __init__(self, corner_topic, path_topic, bounds, turning_radius, wp_resolution):
        

        # Planner variables
        self.state              = []
        self.planner_initial_pose   = []
        self.max_travel_distance    = rospy.get_param("~max_travel_distance")

        # Parameters for optimizer
        self.wp_resolution      = wp_resolution
        self.bounds             = bounds
        self.turning_radius     = turning_radius
        self.distance_travelled = 0

        # Signal to end survey and save data
        self.survey_finished = False
        finished_top = rospy.get_param("~survey_finished_top", '/survey_finished')
        self.synch_pub = rospy.Subscriber(finished_top, Bool, self.synch_cb)

        # Publish corners locations to produce IPs 
        self.corner_pub  = rospy.Publisher(corner_topic, PointCloud2, queue_size=1, latch=True)
        corners = ipp_utils.generate_ip_corners(self.bounds)
        self.corner_pub.publish(corners)

        tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tfBuffer)
        self.map_frame = rospy.get_param('~map_frame', 'map') # map frame_id
        odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.mbes_frame = rospy.get_param('~mbes_link', 'mbes_link') # mbes frame_id
        self.base_frame = rospy.get_param('~base_link', 'base_link')
        self.tf_listener = tf.TransformListener()

        try:
            rospy.loginfo("Waiting for transforms")
            mbes_tf = tfBuffer.lookup_transform(self.base_frame, self.mbes_frame,
                                                rospy.Time(0), rospy.Duration(35))
            self.T_base_mbes = matrix_from_tf(mbes_tf)

            m2o_tf = tfBuffer.lookup_transform(self.map_frame, odom_frame,
                                               rospy.Time(0), rospy.Duration(35))
            self.T_map_odom = matrix_from_tf(m2o_tf)
            rospy.loginfo("Transforms locked - auv_ui_online node")
        except:
            rospy.logerr("ERROR: Could not lookup transform from base_link to mbes_link")
            return

        self.pose_t = np.array([0., 0., 0., 0., 0., 0.])
        self.odom_init = False
        self.odom_topic = rospy.get_param("~odom_topic")
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_update_cb)

        self.pings_num = 0
        mbes_pings_top = rospy.get_param("~mbes_pings_topic", 'mbes_pings')
        rospy.Subscriber(mbes_pings_top, PointCloud2, self.mbes_cb, queue_size=1)

        # Path publisher - publishes waypoints for AUV to follow
        self.path_pub = rospy.Publisher(path_topic, Path, queue_size=100)
        # Server to provide IPP paths
        bo_replan_as = rospy.get_param("~bo_replan_as")
        self._as_plan = actionlib.SimpleActionServer(bo_replan_as, PathPlanAction,
                                                        execute_cb=self.execute_planning, auto_start=False)
        self._as_plan.start()

        rospy.spin()

    def pcloud2ranges(self, point_cloud):
        fields = point_cloud.fields
        assert len(fields) >= 3
        assert fields[0].name == 'x' and fields[0].datatype == PointField.FLOAT32
        assert fields[1].name == 'y' and fields[1].datatype == PointField.FLOAT32
        assert fields[2].name == 'z' and fields[2].datatype == PointField.FLOAT32

        buf = np.frombuffer(point_cloud.data, dtype=np.float32)
        pts = buf.view(np.uint8).reshape(-1, point_cloud.point_step)
        xyz = pts[:, 0:12].view(np.float32).reshape(-1, 3)

        return xyz

    def synch_cb(self, finished_msg):
        self.survey_finished = finished_msg.data
        rospy.signal_shutdown("It's over bitches")


    def mbes_cb(self, mbes_ping):
        
        if not self.survey_finished and self.odom_init:
            
            T_odom_base = vec2homMat(self.pose_t.T) 
            Tm2mbes = self.T_map_odom @ T_odom_base @ self.T_base_mbes

            # Beams in ping as array in homogeneous coordinates
            beams_mbes = self.pcloud2ranges(mbes_ping)
            beams_mbes = np.hstack((beams_mbes, np.ones((len(beams_mbes), 1))))
            # Transform to map frame
            beams_in_map = (Tm2mbes @ beams_mbes.T)[0:3].T

            # Do your GP training here with beams_in_map
                
            self.pings_num += 1


    def odom_update_cb(self, msg):
        """ Gets our current 2D state (x,y,theta) from the tf frames.
            Also calculates the total distance travelled.

        Args:
            msg (PoseWithCovarianceStamped): A pose message (assumes msg type from AUV messages)
        """

        p = PoseStamped(header=msg.header, pose=msg.pose.pose)
        p.header.stamp = msg.header.stamp
        try:
            self.tf_listener.waitForTransform(self.map_frame, p.header.frame_id, rospy.Time(0), timeout=rospy.Duration(10.)) # 0.1s
            p_in_map = self.tf_listener.transformPose(self.map_frame, p)
            explicit_quat = [p_in_map.pose.orientation.x, p_in_map.pose.orientation.y, p_in_map.pose.orientation.z, p_in_map.pose.orientation.w]
            roll, pitch, yaw = tf.transformations.euler_from_quaternion(explicit_quat)
            self.pose_t = np.array([p_in_map.pose.position.x, p_in_map.pose.position.y, p_in_map.pose.position.z, roll, pitch, yaw])
            if len(self.state) > 0:
                self.distance_travelled += np.hypot(p_in_map.pose.position.x - self.state[0], p_in_map.pose.position.y - self.state[1])
            self.state = [p_in_map.pose.position.x, p_in_map.pose.position.y, yaw]
            
            if self.odom_init == False:
                self.planner_initial_pose   = copy.deepcopy(self.state)
            
            self.odom_init = True
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logdebug("Couldn't get transform")


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
            for pose in wp_poses[skip:]:
                wp = PoseStamped()
                wp.header = h
                wp.pose.position.x = pose[0]
                wp.pose.position.y = pose[1]
                wp.pose.orientation = geometry_msgs.msg.Quaternion(*tf_conversions.transformations.quaternion_from_euler(0, 0, pose[2]))
                sampling_path.poses.append(wp)
            self.planner_initial_pose = wp_poses[-1]
        return sampling_path
    
    
    def execute_planning(self, goal):
        """ 
            This callback essentially runs the entire planning loop
        """
    
        if self.odom_init:

            # 2 possible planning approaches:
            if goal.request == 0:
                rospy.loginfo("Initial deterministic path requested")
                sampling_path = self.initial_deterministic_path()

            else:
                rospy.loginfo("This is where you implement your planner")

                # TODO: publish this trajectory as a set of waypoints
                h = std_msgs.msg.Header()
                h.stamp = rospy.Time.now()
                h.frame_id = self.map_frame
                sampling_path = Path()
                sampling_path.header = h
                print("Current distance travelled: " + str(round(self.distance_travelled)) + " m.")
            
            result = PathPlanResult()
            result.path = sampling_path
            self._as_plan.set_succeeded(result)
            # For visualization in rviz only
            self.path_pub.publish(sampling_path) 

        else:
            print("Odom not initialized yet")
            self._as_plan.set_aborted()


if __name__ == '__main__':

    rospy.init_node('myopic_planner_node' , disable_signals=False)

    # Get parameters from ROS
    turn_radius         = rospy.get_param("~turning_radius")
    corner_topic        = rospy.get_param("~corners_topic")
    path_topic_vis      = rospy.get_param("~path_topic_vis")
    bound_left          = rospy.get_param("~bound_left")
    bound_right         = rospy.get_param("~bound_right")
    bound_up            = rospy.get_param("~bound_up")
    bound_down          = rospy.get_param("~bound_down")
    wp_resolution       = rospy.get_param("~wp_resolution")

    # Make sure map bounds make sense
    low_x = min(bound_left, bound_right)
    high_x = max(bound_left, bound_right)
    low_y = min(bound_down, bound_up)
    high_y = max(bound_down, bound_up)

    bounds = [low_x, low_y, high_x, high_y]
    
    assert bounds[0] < bounds[2],       "planner_node: Given global bounds wrong in X dimension"
    assert bounds[1] < bounds[3],       "planner_node: Given global bounds wrong in Y dimension"
    assert path_topic_vis != "",            "planner_node: Path topic empty"
    assert corner_topic != "",          "planner_node: Corner topic empty"

    try:
        rospy.loginfo("Initializing planner node! Using Bayesian Optimization.")  
        planner = MaxIPPPlaner(corner_topic=corner_topic, path_topic=path_topic_vis, 
                               bounds=bounds, turning_radius=turn_radius, wp_resolution=wp_resolution)
        
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch myopic planner")