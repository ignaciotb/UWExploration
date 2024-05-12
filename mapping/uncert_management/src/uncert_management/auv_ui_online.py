#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Bool
import tf2_ros
from tf.transformations import translation_matrix, quaternion_matrix 
import tf

from sympy import *
import matplotlib.pyplot as plt
import numpy as np
import math
from barfoot_utils import create_rot_sym
from barfoot_utils_np import *
# from auvlib.data_tools import std_data, all_data
from optparse import OptionParser
from scipy.spatial.transform import Rotation as Rot
import os
import actionlib

from slam_msgs.msg import ManipulatePosteriorAction, ManipulatePosteriorGoal
from slam_msgs.msg import SamplePosteriorResult, SamplePosteriorAction
from slam_msgs.msg import MinibatchTrainingAction, MinibatchTrainingResult, MinibatchTrainingGoal
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2

from gpytorch.distributions import MultivariateNormal
import torch

from std_srvs.srv import Empty
from nav_msgs.msg import Path
from scipy.spatial.transform import Rotation
import time 

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


class auv_ui_online(object):

    def __init__(self):

        
        self.path_img = rospy.get_param('~background_img_path', 'default_real_mean_depth.png')
        
        #NOTE: commented this out, the asko dataset doesnt have this
        #self.img = plt.imread(self.path_img)

        self.map_frame = rospy.get_param('~map_frame', 'map') # map frame_id
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.odom_ipp_top = rospy.get_param('~odom_ipp_top', 'odom')
        self.mbes_frame = rospy.get_param('~mbes_link', 'mbes_link') # mbes frame_id
        self.base_frame = rospy.get_param('~base_link', 'base_link')
        self.base_frame_ipp = rospy.get_param('~base_link_ipp', 'base_link')
        self.survey_name = rospy.get_param('~dataset', 'survey')
        
        # Transforms from auv_2_ros
        tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tfBuffer)
        
        # Measurements
        self.yspcov_vec = []
        self.ysp_vec = []
        self.m_vec = []

        # Initial state
        self.mu_t = np.array([0., 0., 0., 0., 0., 0.])
        self.sigma_t = np.diag([0.00001,0.00001,0.000001,0.0000001,0.0000001,0.0000001]) 
        
        self.mu_vec = np.zeros((3, 1))  # For plotting
        self.gt_pose_vec = np.zeros((3, 1))  # For plotting
        self.time = rospy.Time.now().to_sec()
        self.old_time = rospy.Time.now().to_sec()
        self.first_odom = True
        
        # Noise models
        self.Q_3d = np.diag([0.0001, 0.0001, 0.0001]) # Meas noise (x,y,z)
        # self.Q_sens = np.diag([0.01, 0.1, 0.1]) # Meas noise (range, bearing, along track)
        self.R = np.diag([0.000000,0.000000,0.00000,0.00000,0.00000,0.000000005]) # Motion noise

        try:
            rospy.loginfo("Waiting for transforms")
            mbes_tf = tfBuffer.lookup_transform(self.base_frame, self.mbes_frame,
                                                rospy.Time(0), rospy.Duration(35))
            self.T_base_mbes = matrix_from_tf(mbes_tf)

            m2o_tf = tfBuffer.lookup_transform(self.map_frame, self.odom_frame,
                                               rospy.Time(0), rospy.Duration(35))
            self.T_map_odom = matrix_from_tf(m2o_tf)
            rospy.loginfo("Transforms locked - auv_ui_online node")
        except:
            rospy.loginfo("ERROR: Could not lookup transform from base_link to mbes_link")

        ## Symbols
        # State
        x, y, z, rho, phi, theta = symbols('x y z rho phi theta', real=True)
        # Control input
        vx, vy, vz, wx, wy, wz = symbols('vx vy vz wx wy wz', real=True)
        dt = Symbol('dt', real=True)
        # Vehicle state
        self.X = Matrix([x,y,z, rho, phi, theta])
        V = Matrix([vx,vy,vz, wx, wy, wz])

        # Rotation matrices
        self.Rxyz = create_rot_sym(self.X[3:6])

        # 3D motion model and Jacobian
        self.g = sym.lambdify([self.X, V, dt], self.motion_model(self.X,V,dt), "numpy")
        self.G = sym.lambdify([self.X, V, dt], self.motion_model(self.X, V, dt).jacobian(self.X), "numpy")

        # Signal to end survey and save data
        finished_top = rospy.get_param("~survey_finished_top", '/survey_finished')
        self.synch_pub = rospy.Subscriber(finished_top, Bool, self.synch_cb)
        self.survey_finished = False
        self.start_training = False
        # self.covs_all = []
        # self.means_all = []
        self.covs_all = np.empty([1,2,2])
        self.means_all = np.empty([1, 3])
        self.jitter = np.identity(2) * 1e-6

        self.pings_num = 0
        self.pings_num_prev = 0
        self.save_img = False

        # For testing IPP
        manipulate_gp_name = rospy.get_param("~manipulate_gp_server")
        self.ac_manipulate = actionlib.SimpleActionClient(manipulate_gp_name, ManipulatePosteriorAction)
        while not self.ac_manipulate.wait_for_server(timeout=rospy.Duration(5)) and not rospy.is_shutdown():
            print("Waiting for Manipulate AS ")

        # sample_gp_name = rospy.get_param("~sample_gp_server")
        # self.ac_sample = actionlib.SimpleActionClient(sample_gp_name, SamplePosteriorAction)
        # while not self.ac_sample.wait_for_server(timeout=rospy.Duration(5)) and not rospy.is_shutdown():
        #     print("Waiting for sample AS ")

        # Send to as and wait
        # ac_sample.send_goal(goal)
        # ac_sample.wait_for_result()
        # result = ac_sample.get_result()
        # Sample GP with the ping in map frame
        # mu, sigma = result.mu, result.sigma

        # AS for minibath training data from RBPF
        mb_gp_name = rospy.get_param("~minibatch_gp_server")
        self.mb_server = actionlib.SimpleActionServer(mb_gp_name, MinibatchTrainingAction,
                                                execute_cb=self.mb_cb, auto_start = False)
        self.mb_server.start()

        ip_top = rospy.get_param("~inducing_points_top")
        self.ip_pub = rospy.Publisher(ip_top, PointCloud2, queue_size=10)

        self.test_odom = rospy.Publisher(self.odom_ipp_top, Odometry, queue_size=10)

        # Subscription to path to distribute inducing points
        wp_top = rospy.get_param("~corners_topic", "/corners")
        rospy.Subscriber(wp_top, Path, self.path_cb, queue_size=1)

        # Subscribe when ready
        odom_top = rospy.get_param("~odometry_topic", 'odom')
        rospy.Subscriber(odom_top, Odometry, self.odom_cb, queue_size=1)
        
        mbes_pings_top = rospy.get_param("~mbes_pings_topic", 'mbes_pings')
        rospy.Subscriber(mbes_pings_top, PointCloud2, self.mbes_cb, queue_size=1)

        # For gaussian sampling
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        rospy.spin()

    
    def path_cb(self, path):

        if len(path.poses) > 0:

            if not self.start_training:

                rospy.loginfo("Sending inducing points")

                wps = []
                for pose in path.poses:
                    print("X: " + str(pose.pose.position.x) + " Y: " + str(pose.pose.position.y))
                    wps.append(np.array([pose.pose.position.x, pose.pose.position.y, 0.]))

                wp_cloud = pack_cloud(self.map_frame, wps)
                print(wp_cloud)
                self.ip_pub.publish(wp_cloud)

                # This service will start the auv simulation or auv_2_ros nodes to start the mission
                synch_top = rospy.get_param("~synch_topic", '/pf_synch')
                self.srv_server = rospy.Service(synch_top, Empty, self.empty_srv)

                self.start_training = True


    def empty_srv(self, req):
        rospy.loginfo("AUV UI online ready")
        return None


    def mb_cb(self, goal):
        t = time.time()
        mb_size = goal.mb_size
        result = MinibatchTrainingResult()

        if not self.survey_finished and self.means_all.shape[0] > mb_size:

            # TODO: this is pretty slow as it is            
            # Randomly sample minibatch UIs from current dataset
            idx = np.random.choice(self.means_all.shape[0]-1, mb_size, replace=False)

            # Sample UIs covariances
            inputs = torch.from_numpy(self.means_all[idx, 0:2]).to(self.device).float()
            covs = torch.from_numpy(self.covs_all[idx]).to(self.device).float()
            inputs = MultivariateNormal(inputs, covs).rsample()
            inputs = np.reshape(inputs.cpu().numpy(), (-1, 2))

            # Pack cloud
            cloud = np.hstack((inputs, self.means_all[idx, 2].reshape(-1,1)))
            mbes_pcloud = pack_cloud(self.map_frame, cloud)

            result.minibatch = mbes_pcloud
            result.success = True
            self.mb_server.set_succeeded(result)
            
            # print("Time ", time.time()- t)

        else:
            result.success = False
            self.mb_server.set_succeeded(result)



    def synch_cb(self, finished_msg):
        rospy.loginfo("AUV ui node: Survey finished received. Wrapping up")
        # self.save_img = True
        # rospy.sleep(3)
        self.survey_finished = finished_msg.data

        # Send to as and wait
        goal = ManipulatePosteriorGoal()
        mbes_pcloud = pack_cloud(self.map_frame, np.asarray(self.means_all))
        goal.pings = mbes_pcloud
        goal.sample = False
        goal.plot = self.survey_finished
        self.ac_manipulate.send_goal(goal)
        self.ac_manipulate.wait_for_result()

        np.savez(self.survey_name+"_svgp_input"+".npz", points=self.means_all,
                covs=self.covs_all)
        # np.save(self.survey_name+ "_svgp_input_dr.npy", self.means_all)

        # duration = 2  # seconds
        # freq = 340  # Hz
        # os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
        
        print("Final AUV sigma")
        print(self.sigma_t)
        rospy.signal_shutdown("It's over bitches")


    # Base pose on odom frame
    def odom_cb(self, odom_msg):

        if not self.survey_finished and self.start_training:

            self.time = odom_msg.header.stamp.to_sec()
            
            if self.first_odom:
                self.old_time = self.time
                self.first_odom = False
                return

            dt_real = self.time - self.old_time 
            #print(dt_real)
            # dt_real = 0.2
            
            vt = np.array([odom_msg.twist.twist.linear.x,
                        odom_msg.twist.twist.linear.y,
                        odom_msg.twist.twist.linear.z,
                        #    odom_msg.twist.twist.angular.x + np.random.normal(0, 0.001, 1),
                        odom_msg.twist.twist.angular.x,
                        odom_msg.twist.twist.angular.y,
                           odom_msg.twist.twist.angular.z], dtype=object)
                        # odom_msg.twist.twist.angular.z + np.random.normal(0, 0.002, 1)[0]], dtype=object)

            ## Prediction
            mu_hat_t = np.concatenate(self.g(self.mu_t, vt, dt_real), axis=0)
            for i in range(3,6): # Wrap angles
                mu_hat_t[i] = (mu_hat_t[i] + np.pi) % (2 * np.pi) - np.pi
            
            ## GT for testing
            # Nacho: comment out for Lolo experiments
            # print("-----")
            # quaternion = (odom_msg.pose.pose.orientation.x,
            #                 odom_msg.pose.pose.orientation.y,
            #                 odom_msg.pose.pose.orientation.z,
            #                 odom_msg.pose.pose.orientation.w)
            # euler = tf.transformations.euler_from_quaternion(quaternion)
            
            # self.pose_t = np.array([odom_msg.pose.pose.position.x,
            #                 odom_msg.pose.pose.position.y,
            #                 odom_msg.pose.pose.position.z,
            #                 euler[0],
            #                 euler[1],
            #                 euler[2]])
            # for i in range(3,6): # Wrap angles
            #     self.pose_t[i] = (self.pose_t[i] + np.pi) % (2 * np.pi) - np.pi
            # print(self.pose_t - mu_hat_t)
            
            Gt = np.concatenate(np.array(self.G(self.mu_t, vt, dt_real)).astype(np.float64), 
                                axis=0).reshape(6,6)
            sigma_hat_t = Gt @ self.sigma_t @ Gt.T + self.R

            ## Update
            self.mu_t = mu_hat_t
            self.sigma_t = sigma_hat_t
            # print('\n'.join([''.join(['{:4}'.format(item) for item in row])
            #                  for row in self.sigma_t]))
            # print(self.mu_t)

            # For testing
            quat = quaternion_from_euler(self.mu_t[3], self.mu_t[4], self.mu_t[5], "rxyz")
            odom_msg = Odometry()
            odom_msg.header.stamp = odom_msg.header.stamp
            odom_msg.header.frame_id = self.odom_frame
            odom_msg.child_frame_id = "lolo/base_link_ipp"
            odom_msg.pose.covariance = [0.] * 36
            odom_msg.pose.pose.position.x = self.mu_t[0]
            odom_msg.pose.pose.position.y = self.mu_t[1]
            odom_msg.pose.pose.position.z = self.mu_t[2]
            odom_msg.pose.pose.orientation.x = quat[0]
            odom_msg.pose.pose.orientation.y = quat[1]
            odom_msg.pose.pose.orientation.z = quat[2]
            odom_msg.pose.pose.orientation.w = quat[3]
            self.test_odom.publish(odom_msg)

            self.old_time = self.time

        
    def mbes_cb(self, mbes_ping):
        
        if not self.survey_finished and self.start_training:
            
            # Measurements
            self.yspcov_vec.clear()
            self.ysp_vec.clear()
            self.m_vec.clear()
            
            # T map to mbes and compound cov at time t
            T_odom_base = vec2homMat(self.mu_t.T) 
            Tm2mbes = self.T_map_odom @ T_odom_base @ self.T_base_mbes
            Covt = self.compound_covs(self.sigma_t, self.Q_3d)

            # Ping as array in homogeneous coordinates
            beams_mbes = pcloud2ranges_full(mbes_ping)
            # print("-----------------")
            # print(beams_mbes)
            beams_mbes = np.hstack((beams_mbes, np.ones((len(beams_mbes), 1))))

            # Use only N beams
            N = 50
            idx = np.round(np.linspace(0, len(beams_mbes)-1, N)).astype(int)
            beams_mbes_filt = beams_mbes[idx]
            #print("UI ping ", self.pings_num, " with: ", len(beams_mbes_filt), " beams")
            
            for n, beam in enumerate(beams_mbes_filt):
            # for n in range(len(beams_mbes)):
                # Create landmark as expected patch of seabed to be hit (in map frame)
                beam_map = np.matmul(Tm2mbes, beam)
                # beam_map = np.matmul(Tm2mbes, beams_mbes[n])
                # print(beam_map[0:3])
            
                # Expected measurement to beam_map as 3D coordinates
                ht_3d = self.meas_model_3D(Tm2mbes, Tm2mbes, beam_map)
                
                ## Sigma points cov calculation at time t in vehicle domain
                ysp, yspcov = self.sigmapoints_cov(Tm2mbes, Covt, beam_map, ht_3d)
                
                # Save results for plotting
                # self.ysp_vec.append(ysp)
                # self.yspcov_vec.append(yspcov[0:3,0:3])
                # self.m_vec.append(beam_map[0:3])

                # Store real MBES beams and approximated covariances
                # self.covs_all.append(yspcov[0:2,0:2])
                # self.means_all.append(beam_map[0:3])
                if not np.all(np.linalg.eigvals(yspcov[0:2, 0:2]) > 0.):
                    rospy.logwarn("Non PD cov matrix")
                    yspcov[0:2, 0:2] += self.jitter
                
                self.means_all = np.append(self.means_all, beam_map[0:3].reshape(-1, 3), axis=0)
                self.covs_all =np.append(self.covs_all, yspcov[0:2, 0:2].reshape(1,2,2), axis=0)

            # Remove initilizations vals from arrays
            if self.pings_num == 0:
                self.means_all = self.means_all[1:]
                self.covs_all = self.covs_all[1:]
                
            self.pings_num += 1
            # # Plotting
            # self.visualize()

    # Motion model in 3D
    def motion_model(self, X, V, dt):
        g_r = Matrix([X[3:6]]) + Matrix([V[3:6]]).multiply(dt)
        g_p = Matrix(X[0:3]) + self.Rxyz.subs([(X[3], X[3]),
                                               (X[4], X[4]), 
                                               (X[5], X[5])]).multiply(
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
    

    def visualize(self, save=False):
           
        ## Plotting    
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.cla()

        # Transform to odom frame before plotting
        # Overnight 20
        # plt.imshow(self.img, extent=[-647-self.T_map_odom[0,3], 1081-self.T_map_odom[0,3],
        #                              -1190-self.T_map_odom[1,3], 523-self.T_map_odom[1,3]])
        # plt.axis([-250, 650, -100, 600])

        plt.imshow(self.img, extent=[-984-self.T_map_odom[0,3], 193-self.T_map_odom[0,3],
                                     -667-self.T_map_odom[1,3], 1175-self.T_map_odom[1,3]])

        #  mu_t_np = np.array(self.mu_t[0:3]).astype(np.float64)
        mu_t = self.T_map_odom[0:3,0:3].dot(self.mu_t[0:3])
        self.mu_vec = np.hstack((self.mu_vec, mu_t.reshape(3,1)))
        
        # Plot mut, sigmat, sigma points and mc 
        plt.plot(self.mu_vec[0, :],
                 self.mu_vec[1, :], "-r")

        # Plot ground truth pose
        gt_pose_t = self.T_map_odom[0:3, 0:3].dot(self.pose_t[0:3])
        self.gt_pose_vec = np.hstack(
            (self.gt_pose_vec, gt_pose_t.reshape(3, 1)))
        
        # Plot mut, sigmat, sigma points and mc 
        plt.plot(self.gt_pose_vec[0, :],
                 self.gt_pose_vec[1, :], "-b")


        # Local copies for plotting
        ysp_vec_plot = self.ysp_vec
        yspcov_vec_plot = self.yspcov_vec
        m_vec_plot = self.m_vec
        N = len(m_vec_plot)

        # for n in range(N):
        #     # Plot sigma points means
        #     # plt.plot(ysp_vec_plot[n][0],
        #     #          ysp_vec_plot[n][1], "+")
        #     # Plot real mbes hits
        #     plt.plot(m_vec_plot[n][0],
        #              m_vec_plot[n][1], "x")
        
        # # Covariances of motion and sigma points
        cov_mat = (self.T_map_odom[0:3,0:3].transpose().dot(self.sigma_t[0:3,0:3])).dot(self.T_map_odom[0:3,0:3])
        px, py = self.plot_covariance(mu_t, cov_mat, 5)
        plt.plot(px, py, "--r")

        # for n in range(N):
        #     px, py = self.plot_covariance(ysp_vec_plot[n], yspcov_vec_plot[n], 5)
        #     #  print(yspcov_vec_plot[n])
        #     plt.plot(px, py, "--g")

        plt.grid(True)
        plt.tight_layout()
        plt.pause(0.00001)

        if save:
            print("Saving final plot")
            plt.savefig(self.survey_name + "_survey.png")


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
        rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
        fx = np.stack([x, y]).T @ rot

        px = np.array(fx[:, 0] + xEst[0]).flatten()
        py = np.array(fx[:, 1] + xEst[1]).flatten()

        return px, py

if __name__ == '__main__':

    rospy.init_node('auv_ui_online', disable_signals=True)
    try:
        auv_ui_online()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not launch auv ui')








