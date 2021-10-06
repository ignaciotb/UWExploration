#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Bool
import tf2_ros
from tf.transformations import translation_matrix, quaternion_matrix 

from sympy import *
import matplotlib.pyplot as plt
import numpy as np
import math
from barfoot_utils import create_rot_sym
from barfoot_utils_np import *
from auvlib.data_tools import std_data, all_data
from optparse import OptionParser
from scipy.spatial.transform import Rotation as Rot


def pcloud2ranges_full(point_cloud):
    ranges = []
    for p in pc2.read_points(point_cloud, 
                             field_names = ("x", "y", "z"), skip_nans=True):
        ranges.append(p)

    return np.asarray(ranges)


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


class auv_ui(object):

    def __init__(self):

        
        self.path_img = rospy.get_param('~background_img_path', 'default_real_mean_depth.png')
        self.img = plt.imread(self.path_img)

        self.map_frame = rospy.get_param('~map_frame', 'map') # map frame_id
        self.mbes_frame = rospy.get_param('~mbes_link', 'mbes_link') # mbes frame_id
        odom_frame = rospy.get_param('~odom_frame', 'odom')

        self.survey_name = rospy.get_param('~survey_name', 'survey')
        
        # Transforms from auv_2_ros
        tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tfBuffer)
        
        # Measurements
        self.yspcov_vec = []
        self.ysp_vec = []
        self.m_vec = []

        # Initial state
        self.mu_t = np.array([0., 0., 0., 0., 0., 0.])
        self.sigma_t = np.diag([0.0001,0.0001,0.0001,0.000001,0.000001,0.00001])
        self.mu_vec = np.zeros((3,1)) # For plotting
        self.time = rospy.Time.now().to_sec()
        self.old_time = rospy.Time.now().to_sec()
        
        # Noise models
        self.Q_3d = np.diag([0.00001, 0.00001, 0.00001]) # Meas noise (x,y,z)
        self.Q_sens = np.diag([0.01, 0.1, 0.1]) # Meas noise (range, bearing, along track)
        self.R = np.diag([0.000001,0.000001,0.000001,0.000001,0.000001,0.00001]) # Motion noise

        try:
            rospy.loginfo("Waiting for transforms")
            mbes_tf = tfBuffer.lookup_transform('hugin/base_link', 'hugin/mbes_link',
                                                rospy.Time(0), rospy.Duration(35))
            self.T_base_mbes = matrix_from_tf(mbes_tf)

            m2o_tf = tfBuffer.lookup_transform(self.map_frame, odom_frame,
                                               rospy.Time(0), rospy.Duration(35))
            self.T_map_odom = matrix_from_tf(m2o_tf)

            rospy.loginfo("Transforms locked - auv_ui node")
        except:
            rospy.loginfo("ERROR: Could not lookup transform from base_link to mbes_link")

        ## Symbols
        # State
        x, y, z, rho, phi, theta = symbols('x y z rho phi theta', real=True)
        xn, yn, zn, rhon, phin, thetan = symbols('xn yn zn rhon phin thetan', real=True)
        # Landmark
        mx, my, mz = symbols('mx my mz', real=True)
        # Control input
        vx, vy, vz, wx, wy, wz = symbols('vx vy vz wx wy wz', real=True)
        dt = Symbol('dt', real=True)

        # For the MC 
        rng = np.random.default_rng()

        # Vehicle state and landmark
        self.X = Matrix([x,y,z, rho, phi, theta])
        Xn = Matrix([xn,yn,zn, rhon, phin, thetan])
        V = Matrix([vx,vy,vz, wx, wy, wz])
        m = Matrix([mx, my, mz])

        # Rotation matrices
        self.Rxyz = create_rot_sym(self.X[3:6])

        # 3D motion model and Jacobian
        self.g = sym.lambdify([self.X, V, dt], self.motion_model(self.X,V,dt), "numpy")
        self.G = sym.lambdify([self.X,V,dt], self.motion_model(self.X, V, dt).jacobian(self.X), "numpy")

        # Signal to end survey and save data
        finished_top = rospy.get_param("~survey_finished_top", '/survey_finished')
        self.synch_pub = rospy.Subscriber(finished_top, Bool, self.synch_cb)
        self.survey_finished = False
        self.covs_all = []
        self.beams_all = []

        # Subscribe when ready
        odom_top = rospy.get_param("~odometry_topic", 'odom')
        rospy.Subscriber(odom_top, Odometry, self.odom_cb)
        
        mbes_pings_top = rospy.get_param("~mbes_pings_topic", 'mbes_pings')
        rospy.Subscriber(mbes_pings_top, PointCloud2, self.mbes_cb)

        rospy.spin()

    def synch_cb(self, finished_msg):
        self.survey_finished = finished_msg.data
        np.savez(self.survey_name+"_svgp_input"+".npz", points=self.beams_all,
                covs=self.covs_all)
        rospy.loginfo("AUV ui node: Survey finished received")
        rospy.signal_shutdown("It's over bitches")

        
    def mbes_cb(self, mbes_ping):
        # Measurements
        self.yspcov_vec.clear()
        self.ysp_vec.clear()
        self.m_vec.clear()
        
        # T map to mbes and compound cov at time t
        T_odom_base = vec2homMat(self.mu_t.T) 
        Tm2mbes = np.matmul(np.matmul(self.T_map_odom, T_odom_base), self.T_base_mbes)
        Covt = self.compound_covs(self.sigma_t, self.Q_3d)

        # Ping as array in homogeneous coordinates
        beams_mbes = pcloud2ranges_full(mbes_ping)
        beams_mbes = np.hstack((beams_mbes, np.ones((len(beams_mbes), 1))))

        # Use only N beams
        N = 40
        idx = np.round(np.linspace(0, len(beams_mbes)-1, N)).astype(int)
        beams_mbes_filt = beams_mbes[idx]
        print("Covs for N beams")
        for n in range(N):
            # Create landmark as expected patch of seabed to be hit (in map frame)
            beam_map = np.matmul(Tm2mbes, beams_mbes_filt[n])
        
            # Expected measurement to beam_map as 3D coordinates
            ht_3d = self.meas_model_3D(Tm2mbes, Tm2mbes, beam_map)
            
            ## Sigma points cov calculation at time t in vehicle domain
            ysp, yspcov = self.sigmapoints_cov(Tm2mbes, Covt, beam_map, ht_3d)
            
            # Save results for plotting
            self.ysp_vec.append(ysp)
            self.yspcov_vec.append(yspcov[0:3,0:3])
            self.m_vec.append(beam_map[0:3])

            # Store data for saving on disk
            self.covs_all.append(yspcov[0:3,0:3])
            self.beams_all.append(beam_map[0:3])

        # # Plotting
        self.visualize()
    
    def visualize(self):
           
        ## Plotting    
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.cla()

        # Transform to odom frame before plotting
        plt.imshow(self.img, extent=[-647-self.T_map_odom[0,3], 1081-self.T_map_odom[0,3],
                                     -1190-self.T_map_odom[1,3], 523-self.T_map_odom[1,3]])

        #  mu_t_np = np.array(self.mu_t[0:3]).astype(np.float64)
        mu_t = self.T_map_odom[0:3,0:3].dot(self.mu_t[0:3])
        self.mu_vec = np.hstack((self.mu_vec, mu_t.reshape(3,1)))
        # Plot mut, sigmat, sigma points and mc 
        plt.plot(self.mu_vec[0, :],
                 self.mu_vec[1, :], "-k")

        # Plot sigma points means
        N = len(self.ysp_vec)
        for n in range(N):
            plt.plot(self.ysp_vec[n][0],
                     self.ysp_vec[n][1], "+")
        
        # Motion, sigma points
        cov_mat = (self.T_map_odom[0:3,0:3].transpose().dot(self.sigma_t[0:3,0:3])).dot(self.T_map_odom[0:3,0:3])
        px, py = self.plot_covariance(mu_t, cov_mat, 6)
        plt.plot(px, py, "--r")
        for n in range(N):
            px, py = self.plot_covariance(self.ysp_vec[n], self.yspcov_vec[n], 6)
            plt.plot(px, py, "--g")

        plt.grid(True)
        plt.pause(0.00001)


    def plot_covariance(self, xEst, C, k):  # pragma: no cover
        eig_val, eig_vec = np.linalg.eig(C[0:2,0:2])

        if eig_val[0] >= eig_val[1]:
            big_ind = 0
            small_ind = 1
        else:
            big_ind = 1
            small_ind = 0

        t = np.arange(0, 2 * math.pi + 0.1, 0.1)

        # eig_val[big_ind] or eiq_val[small_ind] were occasionally negative
        # numbers extremely close to 0 (~10^-20), catch these cases and set
        # the respective variable to 0
        try:
            a = k * math.sqrt(eig_val[big_ind])
        except ValueError:
            a = 0

        try:
            b = k * math.sqrt(eig_val[small_ind])
        except ValueError:
            b = 0

        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        angle = math.atan2(eig_vec[1, big_ind], eig_vec[0, big_ind])
        rot = Rot.from_euler('z', angle).as_dcm()[0:2, 0:2]
        fx = np.stack([x, y]).T @ rot

        px = np.array(fx[:, 0] + xEst[0]).flatten()
        py = np.array(fx[:, 1] + xEst[1]).flatten()

        return px, py


    # Base pose on odom frame
    def odom_cb(self, odom_msg):
        self.time = odom_msg.header.stamp.to_sec()
        dt_real = self.time - self.old_time 
        
        vt = np.array([odom_msg.twist.twist.linear.x,
                       odom_msg.twist.twist.linear.y,
                       odom_msg.twist.twist.linear.z,
                       odom_msg.twist.twist.angular.x,
                       odom_msg.twist.twist.angular.y,
                       odom_msg.twist.twist.angular.z])

        ## Prediction
        mu_hat_t = np.concatenate(self.g(self.mu_t, vt, dt_real), axis=0)
                
        # Sigma hat
        Gt = np.concatenate(np.array(self.G(self.mu_t, vt, dt_real)).astype(np.float64), 
                            axis=0).reshape(6,6)
        sigma_hat_t = np.matmul(np.matmul(Gt, self.sigma_t), Gt.transpose()) + self.R

        # We leave the update step here for now
        ## Update
        for i in range(3,5): # Wrap angles
            mu_hat_t[i] = (mu_hat_t[i] + np.pi) % (2 * np.pi) - np.pi
        self.mu_t = mu_hat_t
        self.sigma_t = sigma_hat_t

        self.old_time = self.time

    # Motion model in 3D
    def motion_model(self, X, V, dt):
        g_r = Matrix([X[3:6]]) + Matrix([V[3:6]]).multiply(dt)
        g_p = Matrix(X[0:3]) + self.Rxyz.subs([(X[3], g_r[0]),
                                          (X[4], g_r[1]), 
                                          (X[5], g_r[2])]).multiply(
                                              Matrix([V[0:3]]).T).multiply(dt)
        g = Matrix(BlockMatrix([[g_p], [g_r.T]]))
        
        return g

    # MBES meas model in 3D: z = [x,y,z] in map frame
    def meas_model_3D(self, T, Tnoisy, mnoisy):
        Tinv_noisy = transInv(Tnoisy)
        h_auv = np.matmul(Tinv_noisy,mnoisy)
        h_map = np.matmul(T, h_auv)

        map_mat = np.array([[1,0,0,0],
                              [0,1,0,0],
                              [0,0,1,0],
                              [0,0,0,0]])

        h = np.matmul(map_mat, h_map)
        return h_map


    def sigmapoints_cov(self, T, Cov, m, ylin):
        Q = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]])
        kappa = 0.
        L = 9

        # Make sure Cov remains positive-semidefinite
        Cov = np.where(Cov>=0., Cov,0.)
        S = np.linalg.cholesky(Cov)
        spoint = np.block([np.zeros((L, 1)),
                           S * np.sqrt(L+kappa),
                           -S * np.sqrt(L+kappa)])
        yspsample = []
        yspsample.append(ylin)
        ysp = yspsample[0] * (kappa/(kappa+L))

        # Sampled mean
        for n in range(1,2*L+1):
            Tsample = np.matmul(vec2tran(spoint[0:6,n]), T)
            msample = m + np.matmul(Q, spoint[6:9,n])
            h = self.meas_model_3D(T, Tsample, msample)

            yspsample.append(h)
            ysp += yspsample[n] * (1/(2*(L+kappa)))

        # Sampled covariance
        yspcov = np.matmul((yspsample[0] - ysp).reshape(4,1), (yspsample[0]-ysp).reshape(1,4)) * (kappa/(kappa + L))
        for n in range(1,2*L+1):
            yspcov += np.matmul((yspsample[n]-ysp).reshape(4,1), (yspsample[n]-ysp).reshape(1,4)) * (1/(2*(kappa + L)))

        return ysp, yspcov


    def compound_covs(self, sigma, Q):
        return np.block([[sigma, np.zeros((6,3))], [np.zeros((3,6)), Q]])
    

if __name__ == '__main__':

    rospy.init_node('auv_ui', disable_signals=True)
    try:
        auv_ui()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not launch auv ui')








