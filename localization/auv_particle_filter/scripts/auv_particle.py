#!/usr/bin/env python3

# Standard dependencies
import math
import rospy
import numpy as np
from scipy.stats import multivariate_normal
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.transform import Rotation as rot

from geometry_msgs.msg import Pose, PoseStamped
from geometry_msgs.msg import Quaternion, Transform

from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf.transformations import translation_matrix, translation_from_matrix
from tf.transformations import quaternion_matrix, quaternion_from_matrix
from tf.transformations import rotation_matrix, rotation_from_matrix

from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2


class Particle(object):
    def __init__(self, beams_num, p_num, index, mbes_tf_matrix, m2o_matrix,
                 init_cov=[0.,0.,0.,0.,0.,0.], meas_cov=0.01,
                 process_cov=[0.,0.,0.,0.,0.,0.]):

        self.p_num = p_num
        self.index = index

        self.beams_num = beams_num
        # self.weight = 1.
        self.p_pose = [0.]*6
        self.mbes_tf_mat = mbes_tf_matrix
        self.m2o_tf_mat = m2o_matrix
        self.init_cov = init_cov
        self.meas_cov = meas_cov
        self.process_cov = np.asarray(process_cov)
        self.w = 0.
        self.log_w = 0.

        self.add_noise(init_cov)

    def add_noise(self, noise):
        noise_cov =np.diag(noise)
        current_pose = np.asarray(self.p_pose)
        noisy_pose = current_pose + np.sqrt(noise_cov).dot(np.random.randn(6,1)).T
        self.p_pose = noisy_pose[0]

    def motion_pred(self, odom_t, dt):
        # Generate noise
        noise_vec = (np.sqrt(self.process_cov)*np.random.randn(1, 6)).flatten()

        # Angular motion
        vel_rot = np.array([odom_t.twist.twist.angular.x,
                            odom_t.twist.twist.angular.y,
                            odom_t.twist.twist.angular.z])

        rot_t = np.array(self.p_pose[3:6]) + vel_rot * dt + noise_vec[3:6]

        roll_t = ((rot_t[0]) + 2 * np.pi) % (2 * np.pi)
        pitch_t = ((rot_t[1]) + 2 * np.pi) % (2 * np.pi)
        yaw_t = ((rot_t[2]) + 2 * np.pi) % (2 * np.pi)
        self.p_pose[3:6] = [roll_t, pitch_t, yaw_t]

        # Linear motion
        vel_p = np.array([odom_t.twist.twist.linear.x,
                         odom_t.twist.twist.linear.y,
                         odom_t.twist.twist.linear.z])

        rot_mat_t = rot.from_euler("xyz", [roll_t,pitch_t, yaw_t]).as_dcm()
        step_t = np.matmul(rot_mat_t, vel_p * dt) + noise_vec[0:3]

        self.p_pose[0] += step_t[0]
        self.p_pose[1] += step_t[1]
        # Seems to be a problem when integrating depth from Ping vessel, so we just read it
        self.p_pose[2] = odom_t.pose.pose.position.z

    def compute_weight(self, exp_mbes, real_mbes_ranges, got_result):
        if got_result:
            # Predict mbes ping given current particle pose and m 
            exp_mbes_ranges = self.list2ranges(exp_mbes)

            if len(exp_mbes_ranges) > 0:
                # Before calculating weights, make sure both meas have same length
                idx = np.round(np.linspace(0, len(real_mbes_ranges) - 1,
                                           self.beams_num)).astype(int)
                mbes_meas_sampled = real_mbes_ranges[idx]
                
                # For debugging
                #  print (len(exp_mbes_ranges))
                #  print (len(mbes_meas_sampled))
                #  print (exp_mbes_ranges)
                #  print (mbes_meas_sampled)
                #  print (exp_mbes_ranges - mbes_meas_sampled)

                # Update particle weights
                self.w = self.weight_mv(mbes_meas_sampled, exp_mbes_ranges)
                #  print "MV ", self.w
                #  self.w = self.weight_avg(mbes_meas_sampled, exp_mbes_ranges)
                #  print "Avg ", self.w
                #  self.w = self.weight_grad(mbes_meas_sampled, exp_mbes_ranges)
                #  print "Gradient", self.w
            else:
                self.w = 1.e-50
        else:
            #  rospy.logwarn("Particle did not get meas")
            self.w = 1.e-50


    def weight_grad(self, mbes_meas_ranges, mbes_sim_ranges ):
        if len(mbes_meas_ranges) == len(mbes_sim_ranges):
            grad_meas = np.gradient(mbes_meas_ranges)
            grad_expected = np.gradient(mbes_sim_ranges)
            w_i = multivariate_normal.pdf(grad_expected, mean=grad_meas,
                                          cov=self.meas_cov)
        else:
            rospy.logwarn("missing pings!")
            w_i = 1.e-50
        return w_i
        
    def weight_mv(self, mbes_meas_ranges, mbes_sim_ranges ):
        if len(mbes_meas_ranges) == len(mbes_sim_ranges):
            w_i = multivariate_normal.pdf(mbes_sim_ranges, mean=mbes_meas_ranges,
                                          cov=self.meas_cov)
        else:
            rospy.logwarn("missing pings!")
            w_i = 1.e-50
        return w_i
      
    def weight_avg(self, mbes_meas_ranges, mbes_sim_ranges ):
        if len(mbes_meas_ranges) == len(mbes_sim_ranges):
            #  w_i = 1./self.p_num
            #  for i in range(len(mbes_sim_ranges)):
            w_i = math.exp(-(((mbes_sim_ranges
                               - mbes_meas_ranges)**2).mean())/(2*self.meas_cov))
        else:
            rospy.logwarn("missing pings!")
            #  w_i = 1./self.p_num
            w_i = 1.e-50
        return w_i
    
    def get_p_mbes_pose(self):
        # Find particle's mbes_frame pose 
        t_particle = translation_matrix(self.p_pose[0:3])
        r_particle = rot.from_euler('xyz', self.p_pose[3:6], degrees=False)
        q_particle = quaternion_matrix(r_particle.as_quat())
        mat = np.dot(t_particle, q_particle)
        
        trans_mat = self.m2o_tf_mat.dot(mat.dot(self.mbes_tf_mat))

        self.p = translation_from_matrix(trans_mat)
        angle, direc, point = rotation_from_matrix(trans_mat)
        self.R = rotation_matrix(angle, direc, point)[0:3, 0:3]
        
        return (self.p, self.R)

    # Extract the z coordinate from exp pings (in map frame)
    def list2ranges(self, points):
        ranges = []
        for p in points:
            ranges.append(p[2])

        return np.asarray(ranges)

    
# Extract the z coordinate from real pings (in map frame)
def pcloud2ranges(point_cloud):
    ranges = []
    for p in pc2.read_points(point_cloud, 
                             field_names = ("x", "y", "z"), skip_nans=True):
        ranges.append(p[2])

    return np.asarray(ranges)

# Create PointCloud2 msg out of ping    
def pack_cloud(frame, mbes):
    mbes_pcloud = PointCloud2()
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1)]

    mbes_pcloud = point_cloud2.create_cloud(header, fields, mbes)

    return mbes_pcloud 


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
