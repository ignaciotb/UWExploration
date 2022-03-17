#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PointStamped
from move_base_msgs.msg import MoveBaseFeedback, MoveBaseResult, MoveBaseAction, MoveBaseGoal
import actionlib
import rospy
import tf
import tf2_ros
from std_msgs.msg import Float64, Header, Bool
import math
from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped, Point
import roslaunch

# from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf.transformations import translation_matrix, translation_from_matrix
from tf.transformations import quaternion_matrix, quaternion_from_matrix
# from tf.transformations import rotation_matrix, rotation_from_matrix
from scipy.spatial.transform import Rotation as rot

class BackseatDriver(object):

    # create messages that are used to publish feedback/result
    _feedback = MoveBaseFeedback()
    _result = MoveBaseResult()

    def __init__(self, name):
        self._action_name = name

        #self.heading_offset = rospy.get_param('~heading_offsets', 5.)
        # self.planner_as_name = rospy.get_param('~path_planner_as')
        self.path_topic = rospy.get_param('~path_topic')
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.base_frame = rospy.get_param('~base_frame', 'base_link')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.avg_pose_top = rospy.get_param("~average_pose_topic", '/average_pose')
        self.cov_threshold = rospy.get_param("~cov_threshold", 50)
        self.wp_topic = rospy.get_param('~wp_topic')
        self.goal_tolerance = rospy.get_param('~goal_tolerance', 5.)
        self.enable_pf_update_topic = rospy.get_param("~enable_pf_update_topic")
        self.sim_pf_launch = rospy.get_param('~auv_pf_launch_file', "particle.launch") 
        self.mission_launch = rospy.get_param('~mission_launch_file', "particle.launch") 
        self.sim_path_topic = rospy.get_param('~sim_path_topic')
        self.relocalize_topic = rospy.get_param('~relocalize_topic')

        self.listener = tf.TransformListener()
        
        # To send LC wp to the mission planner
        self.wp_pub = rospy.Publisher(self.wp_topic, PoseStamped, queue_size=1)

        # Stop missiong while choosing revisiting area
        self.pause_mission_pub = rospy.Publisher(self.relocalize_topic, Bool, queue_size=1)

        # The waypoints as a path
        rospy.Subscriber(self.path_topic, Path, self.path_cb, queue_size=1)
        self.latest_path = Path()

        # The PF filter state
        rospy.Subscriber(self.avg_pose_top, PoseWithCovarianceStamped,
                         self.pf_cb, queue_size=1)
        self.closing_loop = False
        self.new_wp = PoseStamped()

        # The SIM PF filter state
        rospy.Subscriber('/pf_sim/pf/avg_pose', PoseWithCovarianceStamped, self.sigma_k_cb)
        self.e_mbes_pub = rospy.Publisher(self.enable_pf_update_topic, Bool, queue_size=1)
        self.sim_path_pub = rospy.Publisher(self.sim_path_topic, Path, queue_size=1)

        # The LC waypoints, as a path
        self.lc_waypoints = Path()
        self.lc_waypoints.header.frame_id = self.map_frame
        # Two LC wp for testing
        lc_wp = PoseStamped()
        lc_wp.header.frame_id = self.map_frame
        lc_wp.pose.position.x = -190.
        lc_wp.pose.position.y = -330.
        lc_wp.pose.position.z = 0.
        lc_wp.pose.orientation.w = 1
        self.lc_waypoints.poses.append(lc_wp)

        lc_wp = PoseStamped()
        lc_wp.header.frame_id = self.map_frame
        lc_wp.pose.position.x = -90.
        lc_wp.pose.position.y = -375.
        lc_wp.pose.position.z = 0.
        lc_wp.pose.orientation.w = 1
        self.lc_waypoints.poses.append(lc_wp)

        # Transforms from auv_2_ros
        tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tfBuffer)
        try:
            rospy.loginfo("Waiting for transforms")
            m2o_tf = tfBuffer.lookup_transform(self.map_frame, self.odom_frame,
                                               rospy.Time(0), rospy.Duration(35))
            self.m2o_mat = matrix_from_tf(m2o_tf)
            rospy.loginfo("Got map to odom")

        except:
            rospy.loginfo("ERROR: Could not lookup transform from base_link to mbes_link")
        # rospy.spin()

        self.trc = 0.
        # This has to be in the main thread to be able to use roslaunch
        while not rospy.is_shutdown():

            print("Current PF trace ", self.trc)
            if self.trc > self.cov_threshold and not self.closing_loop:
                # Pose uncertainty too high, closing the loop to relocalize
                rospy.loginfo("Uncertainty over threshold. Looking for best revisit")

                # Pause current mission after reaching current wp
                self.pause_mission_pub.publish(True)

                # Choose which WP to revisit
                sigmas = []
                gains = []
                for wp in self.lc_waypoints.poses:
                    # Run the simulated PF to that WP and compute expected uncertainty
                    sigma_k = self.sim_sigma(wp)
                    # Compute gain as inverse of distance to area
                    gain_k = 1./self.calculate_gain(wp)    
                    print("Sigma_k ", np.sum(np.diag(sigma_k)))            
                    print("Gain_k ", gain_k)            
                    sigmas.append(np.sum(np.diag(sigma_k)))
                    gains.append(gain_k)

                # Utility WP_k: normalize sigmas and gains and compute utilities
                alpha = 0.5
                sigmas /= np.max(sigmas)
                gains /= np.max(gains)
                u = (1-alpha)* np.asarray(gains) - alpha * np.asarray(sigmas)
                print("Utilities ", u)

                # Choose WP_k with max utility and send to mission planner
                k = max(range(len(u)), key=u.__getitem__)
                print("Chosen revisit area ", k)
                self.new_wp = self.lc_waypoints.poses[k]
                self.wp_pub.publish(self.new_wp)

                # Let mission planner continue
                self.pause_mission_pub.publish(False)
                rospy.loginfo("Sent LC waypoint")
                self.closing_loop = True
            
            rospy.Rate(1).sleep()


    def path_cb(self, path_msg):
        self.latest_path = path_msg
        rospy.loginfo("Path received with number of wp: %d",
                      len(self.latest_path.poses))


    def pf_cb(self, pf_msg):
        # Reconstruct PF pose and covariance 
        self.cov = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                self.cov[i, j] = pf_msg.pose.covariance[i*3 + j]
        self.pf_pose = pf_msg.pose.pose
        
        # Monitor trace. If too high, start revisit place selection in main thread
        self.trc = np.sum(np.diag(self.cov))
        # print("Trace ", self.trc)

        # Closing loop
        if self.closing_loop:
            rospy.loginfo("Going for a loop closure!")
            dist = self.distance_wp_frame(self.new_wp, self.base_frame)
            rospy.loginfo("BS driver diff " + str(dist))
            # Stop loop closure when revisit area reached
            # or filter cov already under threshold
            if dist < self.goal_tolerance or self.trc < self.cov_threshold:
                # Goal reached
                rospy.loginfo("Loop closed!")
                self.closing_loop = False


    def sim_sigma(self, wp_k):
        
        # Launch simulation k with current state of the real PF
        # Transform PF pose to map frame
        quaternion = (self.pf_pose.orientation.x, self.pf_pose.orientation.y, 
                        self.pf_pose.orientation.z, self.pf_pose.orientation.w)
        euler_odom = tf.transformations.euler_from_quaternion(quaternion)
        r_particle = rot.from_euler('xyz', euler_odom, degrees=False)
        q_particle = quaternion_matrix(r_particle.as_quat())
        t_particle = translation_matrix([self.pf_pose.position.x, self.pf_pose.position.y, self.pf_pose.position.z])
        mat = np.dot(t_particle, q_particle)
        pf_pose_map = self.m2o_mat.dot(mat)
        # print(pf_pose_map)
        euler_map = rot.from_matrix(pf_pose_map[0:3, 0:3]).as_euler('xyz')

        cli_args = [self.sim_pf_launch ,'namespace:=pf_sim', 'x:=' + str(pf_pose_map[0,3]), 
        'y:=' + str(pf_pose_map[1,3]), 'z:=' + str(pf_pose_map[2,3]), 
        'roll:=' + str(euler_map[0]), 'pitch:=' + str(euler_map[1]),'yaw:=' + str(euler_map[2]),
        'init_covariance:= ' + str([self.cov[0,0], self.cov[1,1],0.,0.,0.,self.cov[5,5]]), 
        'enable_pf_update:=' + str(False)]
        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        # roslaunch.configure_logging(uuid)
        parent_pf = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
        parent_pf.start()
        rospy.loginfo("Sim PF launched")

        # Launch simulation mission planner
        cli_args = [self.mission_launch ,'namespace:=pf_sim', 'manual_control:=False',
        'max_throttle:=' + str(8.0)]
        roslaunch_args = cli_args[1:]
        roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        # roslaunch.configure_logging(uuid)
        parent_mp = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
        parent_mp.start()
        rospy.loginfo("Sim planner launched")
        rospy.sleep(4)

        # Send WP_k to sim w2w_mission_planner
        sim_path = Path()
        sim_path.poses.append(wp_k)
        self.sim_path_pub.publish(sim_path)

        # Wait for sim AUV PF to reach WP k
        sim_finished = False
        thres_eps = 5.0 # How far from the actual revisit area activate the MBES of the sim filter
        while not rospy.is_shutdown() and not sim_finished:
            # rospy.loginfo("Sim moving to WP!")
            dist = self.distance_wp_frame(wp_k, 'pf_sim/base_link')

            if dist < self.goal_tolerance*thres_eps:
                if thres_eps != 5.0:
                    # Goal reached. Finished simulation
                    sim_finished = True
                    # self.e_mbes_pub.publish(False)

                rospy.loginfo("Sim has reached WP!")
                # When wp_k reached, enable MBES pings and wait 
                self.e_mbes_pub.publish(True)
                thres_eps = 1.0
                
            rospy.Rate(5).sleep()

        # When the PF has converged, store output and kill launch file
        sigma_k = self.cov_k
        print("Sim finished")
        parent_pf.shutdown()
        parent_mp.shutdown()

        return sigma_k

    def sigma_k_cb(self, sim_pf_msg):
        # Reconstruct covariance 
        self.cov_k = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                self.cov_k[i, j] = sim_pf_msg.pose.covariance[i*3 + j]

    def calculate_gain(self, wp_k):

        # Distance from current pose to WP_k
        dist_ab = self.distance_wp_frame(wp_k, self.base_frame)
        
        # Add distance from WP_k to next WP in survey?

        return dist_ab

    def distance_wp_frame(self, wp, frame):
        try:
            (trans, rot) = self.listener.lookupTransform(
                self.map_frame, frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            rospy.logwarn("No transform found ", frame)
            return

        start_pos = np.array(trans)
        end_pos = np.array(
            [wp.pose.position.x, wp.pose.position.y, wp.pose.position.z])

        # Distance from current pose "frame" to wp
        return np.linalg.norm(start_pos - end_pos)

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

if __name__ == '__main__':

    rospy.init_node('backseat_driver')
    bs_driver = BackseatDriver(rospy.get_name())
