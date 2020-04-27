#!/usr/bin/env python 

# Standard dependencies
import sys
import os
import math
import rospy
import numpy as np
import tf
import tf2_ros
import tf_conversions
import tf2_msgs.msg # Not sure if needed
import scipy.stats # For weights
from copy import deepcopy

from geometry_msgs.msg import Pose, PoseArray, PoseWithCovarianceStamped
from geometry_msgs.msg import Quaternion, TransformStamped
from nav_msgs.msg import Odometry
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf.transformations import translation_matrix, translation_from_matrix
from tf.transformations import quaternion_matrix, quaternion_from_matrix

# For sim mbes action client
import actionlib
from auv_2_ros.msg import MbesSimGoal, MbesSimAction
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

# Define
meas_freq = 3 # [Hz] to run mbes_sim


class Particle():
    def __init__(self, index):
        self.index = index # index starts from 1
        self.weight = 1.
        self.pose = Pose()
        self.pose.orientation.w = 1.
        
        if index < 10:
            pcloud_top = '/sim_mbes/particle_0' + str(index)
        else:
            pcloud_top = '/sim_mbes/particle_' + str(index)

        # Initialize sim_mbes pointcloud publisher
        self.pcloud_pub = rospy.Publisher(pcloud_top, PointCloud2, queue_size=1)

    def update(self, vel_vec, noise_vec, dt):
        quat = (self.pose.orientation.x,
                self.pose.orientation.y,
                self.pose.orientation.z,
                self.pose.orientation.w)
        _, _, yaw = euler_from_quaternion(quat)

        """
        NEED TO INCLUDE Z (ALTITUDE) IN UPDATES
        ACTUALLY UP/DOWN ARROWS CHANGE PITCH...
        SOMEHOW PARTICLE POSE (AND MBES READING) NEEDS TO
        CHANGE FROM UP/DOWN ARROW MOVEMENTS IN SIM (I THINK)
        """

        self.pose.position.x += vel_vec[0] * dt * math.cos(yaw) + noise_vec[0] + vel_vec[1] * dt * math.sin(yaw)
        self.pose.position.y += vel_vec[0] * dt * math.sin(yaw) + noise_vec[1] + vel_vec[1] * dt * math.cos(yaw)
        yaw += vel_vec[2] * dt + noise_vec[2] # No need for remainder bc quaternion
        self.pose.orientation = Quaternion(*quaternion_from_euler(0, 0, yaw))


class auv_pf():
    def __init__(self):
        # Read nnecessary ROS parameters
        param = rospy.search_param("map_frame")
        self.map_frame = rospy.get_param(param) # map frame_id
        param = rospy.search_param("odometry_topic")
        self.odom_top = rospy.get_param(param) # odometry msg topic (subscribed)
        param = rospy.search_param("mbes_pings_topic")
        self.mbes_pings_top = rospy.get_param(param) # mbes_pings msg topic (subscribed)
        param = rospy.search_param("particle_poses_topic")
        self.pose_array_top = rospy.get_param(param) # Particle pose array topic (published)
        param = rospy.search_param("particle_count")
        self.pc = rospy.get_param(param) # Particle Count
        
        # Read motion covariance values (and convert to float list) 
        param = rospy.search_param("motion_covariance")
        cov_string = rospy.get_param(param)
        cov_string = cov_string.replace('[','')
        cov_string = cov_string.replace(']','')
        cov_list = list(cov_string.split(", "))
        self.predict_cov = list(map(float, cov_list)) # [xv_cov, yv_cov, yaw_v_cov]

        # Initialize class/callback variables
        self.pred_odom = None
        self.time = None
        self.old_time = None
        self.pos_ = PoseArray()
        self.pos_.header.frame_id = self.map_frame        
        self.mbes_true_pc = None

        # Initialize particle poses publisher
        self.pf_pub = rospy.Publisher(self.pose_array_top, PoseArray, queue_size=10)

        # Initialize tf listener (and broadcaster)
        self.tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(self.tfBuffer)
        # self.broadcaster = tf2_ros.TransformBroadcaster()
                
        try:
            rospy.loginfo("Waiting for transform from base_link to mbes_link")
            mbes_tf = self.tfBuffer.lookup_transform('hugin/mbes_link', 'hugin/base_link', rospy.Time.now(), rospy.Duration(10))
            
            mbes_trans = (mbes_tf.transform.translation.x,
                        mbes_tf.transform.translation.y,
                        mbes_tf.transform.translation.z)
            mbes_quat = (mbes_tf.transform.rotation.x,
                        mbes_tf.transform.rotation.y,
                        mbes_tf.transform.rotation.z,
                        mbes_tf.transform.rotation.w)
            
            tmat_mbes = translation_matrix(mbes_trans)
            qmat_mbes = quaternion_matrix(mbes_quat)
            self.mbes_matrix = np.dot(tmat_mbes, qmat_mbes)
            
            rospy.loginfo("Transform locked from base_link to mbes_link - pf node")

        except:
            rospy.loginfo("ERROR: Could not lookup transform from base_link to mbes_link")
        
        # Initialize list of particles
        self.particles = []
        for i in range(self.pc):
            self.particles.append(Particle(i+1)) # particle index starts from 1

        # Initialize connection to MbesSim action server
        self.ac_mbes = actionlib.SimpleActionClient('/mbes_sim_server',MbesSimAction)
        rospy.loginfo("Waiting for MbesSim action server")
        self.ac_mbes.wait_for_server()
        rospy.loginfo("Connected MbesSim action server")

        # Establish subscription to mbes pings message
        rospy.Subscriber(self.mbes_pings_top, PointCloud2, self.mbes_callback)
        # Establish subscription to odometry message | Last because this will start callback & prediction running
        rospy.Subscriber(self.odom_top, Odometry, self.odom_callback)
        rospy.sleep(0.5) # CAN ADD DURATION INSTEAD? 
        

    def odom_callback(self,msg):
        self.pred_odom = msg
        self.time = self.pred_odom.header.stamp.secs + self.pred_odom.header.stamp.nsecs*10**-9 
        if self.old_time and self.time > self.old_time:
            self.predict()
            self.pub_()
        self.old_time = self.time

    def mbes_callback(self, msg):
        self.mbes_true_pc = msg


    def predict(self):
        # Adding gaussian noice
        pred_noice =  self.process_noise()
        # Unpack odometry message
        dt = self.time - self.old_time
        xv = self.pred_odom.twist.twist.linear.x
        yv = self.pred_odom.twist.twist.linear.y
        # vel = np.sqrt(np.power(xv,2) + np.power(yv,2))
        yaw_v = self.pred_odom.twist.twist.angular.z
        vel_vec = [xv, yv, yaw_v]

        # Update particles pose estimate
        for i in range(len(self.particles)):
            particle = self.particles[i]
            particle.update(vel_vec, pred_noice[i,:], dt)


    def pub_(self):
        self.pos_.poses = []
        for pt in self.particles:
            self.pos_.poses.append(pt.pose)
        # Publish particles with time odometry was received
        self.pos_.header.stamp.secs = int(self.time)
        self.pos_.header.stamp.nsecs = (self.time - int(self.time))*10**9
        self.pf_pub.publish(self.pos_)


    def process_noise(self):
        cov_ = np.zeros((3,3))
        cov_[0,0] = self.predict_cov[0]
        cov_[1,1] = self.predict_cov[1]
        cov_[2,2] = self.predict_cov[2]
        var = np.diagonal(cov_)
        return np.sqrt(var)*np.random.randn(self.pc, 3)


    def measurement(self):
        mbes_meas_ranges = self.pcloud2ranges(self.mbes_true_pc, self.pred_odom.pose)
        """
        Should pred_odom pose not be ussed b/c we won't actually know it ???
        Maybe we should tyake average of partricles here?
        """
        weights = []

        std = 0.01
        """
        No idea what value to use here...
        Should this be a constant or calced
        based on beams / particle poses / etc. ???
        e.g. std = math.sqrt((1/N)*sum((x_i - mu)**2))
        """
        C = len(mbes_meas_ranges)*math.log(math.sqrt(2*math.pi*std**2))

        for particle in self.particles:
            mbes_pcloud = self.pf2mbes(particle)
            mbes_sim_ranges = self.pcloud2ranges(mbes_pcloud, particle)

            try: # Sometimes there is no result for mbes_sim_ranges
                mse = ((mbes_meas_ranges - mbes_sim_ranges)**2).mean()
                # print(particle.index, mse)
            except: # What should we do for reweighting particles without an mbes result???
                print('Caught exception in auv_pf.measurement() function')
                mse = None
            
            """
            Temporary weight calculation
            Replace with something legit
            """
            
            if mse == None:
                particle.weight = 0
            else:
                """
                log(w) = C - (1/(2*std**2))*sum(z_hat - z)**2
                C = #son * log(sqrt(2*pi*std**2))
                """
                particle.weight = math.exp(C - mse/(2*std**2))
            
            particle.weight += 1.e-300 # avoid round-off to zero
            weights.append(particle.weight)

        weights_ = np.asarray(weights)
        self.resample(weights_)

        """
        Calc mean pose of particles and display on RVIZ to compare to true pose
        """

    def resample(self, weights):
        
        # if np.sum(weights) == 0: # Catches situation where all weights go to zero
        #     weights = np.ones(weights.size)
        # Above catch no longer necessary

        # Define cumulative density function
        cdf = np.cumsum(weights)
        cdf /= cdf[cdf.size-1]
        # Multinomial resampling
        r = np.random.rand(self.pc,1)
        indices = []
        for i in range(self.pc):
            indices.append(np.argmax(cdf >= r[i]))
        indices.sort()

        keep = list(set(indices)) # set of particles resampled (independent of count)
        lost = [i for i in range(self.pc) if i not in keep] # particle poses to forget
        dupes = indices[:] # particle poses to replace the forgotten
        for i in keep:
            dupes.remove(i)
            
        if len(lost) > self.pc/2: # Threshold to perform resampling
            for i in range(len(lost)): # Perform resampling
                self.particles[lost[i]].pose = deepcopy(self.particles[dupes[i]].pose)
                """
                Consider adding noise to resampled particle
                """
        else:
            print('Too many particles kept - not resampling')

    def pcloud2ranges(self, point_cloud, particle_):
        ranges = []
        for p in pc2.read_points(point_cloud, field_names = ("x", "y", "z"), skip_nans=True):
            # starts at left hand side of particle's mbes
            dx = particle_.pose.position.x - p[0]
            dy = particle_.pose.position.y - p[1]
            dz = particle_.pose.position.z - p[2]
            dist = math.sqrt((dx**2 + dy**2 + dz**2))
            ranges.append(dist)
        return np.asarray(ranges)


    def pf2mbes(self, particle_):

        # Find particle's mbes pose without broadcasting/listening to tf transforms
        particle_trans = (particle_.pose.position.x,
                        particle_.pose.position.y,
                        particle_.pose.position.z)
        particle_quat = (particle_.pose.orientation.x,
                        particle_.pose.orientation.y,
                        particle_.pose.orientation.z,
                        particle_.pose.orientation.w)

        tmat_part = translation_matrix(particle_trans)
        qmat_part = quaternion_matrix(particle_quat)
        mat_part = np.dot(tmat_part, qmat_part)

        trans_mat = np.dot(mat_part, self.mbes_matrix)

        trans = TransformStamped()
        trans.transform.translation.x = translation_from_matrix(trans_mat)[0]
        trans.transform.translation.y = translation_from_matrix(trans_mat)[1]
        trans.transform.translation.z = translation_from_matrix(trans_mat)[2]
        trans.transform.rotation = Quaternion(*quaternion_from_matrix(trans_mat))


        # Build MbesSimGoal to send to action server
        mbes_goal = MbesSimGoal()
        mbes_goal.mbes_pose.header.frame_id = self.map_frame
        # mbes_goal.mbes_pose.child_frame_id = particle_.mbes_frame_id # The particles will be in a child frame to the map
        mbes_goal.mbes_pose.header.stamp = rospy.Time.now()
        mbes_goal.mbes_pose.transform = trans.transform

        # Get result from action server
        self.ac_mbes.send_goal(mbes_goal)
        # rospy.loginfo("Waiting for MbesSim action Result")
        self.ac_mbes.wait_for_result()
        # rospy.loginfo("Got MbesSim action Result")
        mbes_res = self.ac_mbes.get_result()

        # Pack result into PointCloud2
        mbes_pcloud = PointCloud2()
        mbes_pcloud = mbes_res.sim_mbes
        mbes_pcloud.header.frame_id = self.map_frame

        particle_.pcloud_pub.publish(mbes_pcloud)

        return mbes_pcloud


def main():
    # Initialize ROS node
    rospy.init_node('auv_pf', anonymous=True)
    rospy.loginfo("Successful initilization of node")

    # Create particle filter class
    pf = auv_pf()
    rospy.loginfo("Particle filter class successfully created")

    meas_rate = rospy.Rate(meas_freq) # Hz
    while not rospy.is_shutdown():
        pf.measurement()
        meas_rate.sleep()
    
    # rospy.spin()


if __name__ == '__main__':
    main()

