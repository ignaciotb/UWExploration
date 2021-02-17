#!/usr/bin/env python3

import rospy
import matplotlib.pyplot as plt
import numpy as np
import math

from scipy.spatial.transform import Rotation as Rot
from rospy_tutorials.msg import Floats
from std_msgs.msg import Bool
from rospy.numpy_msg import numpy_msg
import tf2_ros
from rbpf_particle import matrix_from_tf
from sensor_msgs.msg import PointCloud2
import message_filters
import sensor_msgs.point_cloud2 as pc2

class PFStatsVisualization(object):
    
    def __init__(self):
        stats_top = rospy.get_param('~pf_stats_top', 'stats')
        self.stats_sub = rospy.Subscriber(stats_top, numpy_msg(Floats), self.stat_cb)
        self.path_img = rospy.get_param('~background_img_path', 'default_real_mean_depth.png')
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.survey_name = rospy.get_param('~survey_name', 'survey')

        # Real mbes pings subscriber
        mbes_pings_top = rospy.get_param("~mbes_pings_topic", 'mbes_pings')

        # PF ping subscriber
        pf_pings_top = rospy.get_param("~particle_sim_mbes_topic", 'pf_mbes_pings')

        self.real_pings_sub = message_filters.Subscriber(mbes_pings_top, PointCloud2)
        self.pf_pings_sub = message_filters.Subscriber(pf_pings_top, PointCloud2)
        self.ts = message_filters.ApproximateTimeSynchronizer([self.real_pings_sub,
                                                               self.pf_pings_sub],
                                                              20,
                                                              slop=20.0,
                                                              allow_headerless=False)

        self.ts.registerCallback(self.ping_cb)
        self.pings_vec = np.zeros((1,6))

        self.filter_cnt = 1
        self.datagram_size = 17
        self.filt_vec = np.zeros((self.datagram_size,1))
        self.img = plt.imread(self.path_img)

        # Map to odom transform to plot AUV pose on top of image
        tfBuffer = tf2_ros.Buffer()
        tf2_ros.TransformListener(tfBuffer)
        try:
            rospy.loginfo("Waiting for transforms")
            m2o_tf = tfBuffer.lookup_transform(self.map_frame, self.odom_frame,
                                               rospy.Time(0), rospy.Duration(35))
            self.m2o_mat = matrix_from_tf(m2o_tf)
            rospy.loginfo("Transforms locked - stats node")
        except:
            rospy.logerr("Stats node: Could not lookup transforms")
        
        # When the survey is finished, save the data to disk
        finished_top = rospy.get_param("~survey_finished_top", '/survey_finished')
        self.synch_pub = rospy.Subscriber(finished_top, Bool, self.synch_cb)
        self.survey_finished = False

        self.cov_traces = [0.]

        rospy.spin()

    def ping_cb(self, real_ping, pf_ping):
        real_meas = self.ping_to_array(real_ping)
        pf_meas = self.ping_to_array(pf_ping)
        
        idx = np.round(np.linspace(0, np.size(real_meas, 0)-1,
                                   np.size(pf_meas, 0))).astype(int)
        real_meas = real_meas[idx, :]
        self.pings_vec = np.hstack((real_meas, pf_meas))

    def ping_to_array(self, point_cloud):
        ranges = []
        for p in pc2.read_points(point_cloud, 
                                 field_names = ("x", "y", "z"), skip_nans=True):
            ranges.append(p)
        return np.asarray(ranges)

    def synch_cb(self, finished_msg):
        self.survey_finished = finished_msg.data
        np.savez(self.survey_name+".npz", full_dataset=self.filt_vec.tolist())
        rospy.loginfo("Stats node: Survey finished received")


    def plot_covariance_ellipse(self, xEst, PEst):  # pragma: no cover
        #  Pxy = PEst.reshape(3,3)
        cov_mat = np.zeros((3,3))
        cov_mat[np.triu_indices(3, 0)] = np.asarray(PEst).reshape(1,6)
        Pxy = cov_mat[0:2,0:2]
        Pxy[1,0] = Pxy[0,1]
        eig_val, eig_vec = np.linalg.eig(Pxy)

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
            a = math.sqrt(eig_val[big_ind])
        except ValueError:
            a = 0

        try:
            b = math.sqrt(eig_val[small_ind])
        except ValueError:
            b = 0

        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        angle = math.atan2(eig_vec[1, big_ind], eig_vec[0, big_ind])
        rot = Rot.from_euler('z', angle).as_dcm()[0:2, 0:2]
        fx = np.stack([x, y]).T @ rot

        px = np.array(fx[:, 0] + xEst[0]).flatten()
        py = np.array(fx[:, 1] + xEst[1]).flatten()
        plt.plot(px, py, "--r")

   
    def stat_cb(self, stat_msg):

        data_t = stat_msg.data.copy().reshape(self.datagram_size,1)
        # Rotate AUV trajectory to place wrt odom in the image
        data_t[2:5] = self.m2o_mat[0:3,0:3].dot(data_t[2:5])
        data_t[5:8] = self.m2o_mat[0:3,0:3].dot(data_t[5:8])
        data_t[8:11] = self.m2o_mat[0:3,0:3].dot(data_t[8:11])
        
        # Reconstruct 3x3 covariance matrix
        # Not account for z values atm
        cov_mat = np.zeros((3,3))
        cov_mat[np.triu_indices(3, 0)] = np.asarray(data_t[11:17]).reshape(1,6)
        cov_mat[1,0] = cov_mat[0,1]
        cov_mat = (self.m2o_mat[0:3,0:3].transpose().dot(cov_mat)).dot(self.m2o_mat[0:3,0:3])
        data_t[11:17] = cov_mat[np.triu_indices(3)].reshape(6,1)
        self.cov_traces.append(np.trace(cov_mat))

        self.filt_vec = np.hstack((self.filt_vec, data_t))
        self.filter_cnt += 1
        if self.filter_cnt > 0:
            
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            # Plot x,y from GT, odom and PF
            if False:
                plt.cla()
                #  Center image on odom frame
                plt.imshow(self.img, extent=[-647-self.m2o_mat[0,3], 1081-self.m2o_mat[0,3],
                                             -1190-self.m2o_mat[1,3], 523-self.m2o_mat[1,3]])
                #  plt.imshow(self.img, extent=[-740, 980, -690, 1023])
                plt.plot(self.filt_vec[2,:],
                         self.filt_vec[3,:], "-k")

                plt.plot(self.filt_vec[5,:],
                         self.filt_vec[6,:], "-b")

                plt.plot(self.filt_vec[8,:],
                         self.filt_vec[9,:], "-r")

                self.plot_covariance_ellipse(self.filt_vec[5:7,-1], self.filt_vec[11:17,-1])

            # Plot error between DR PF and GT
            if False:
                plt.subplot(3, 1, 1)
                plt.cla()
                plt.plot(np.linspace(0,self.filter_cnt, self.filter_cnt),
                         np.sqrt(np.sum((self.filt_vec[2:4,:]-self.filt_vec[8:10,:])**2,
                                        axis=0)), "-k")
                plt.grid(True)

                # Error between PF and GT
                plt.subplot(3, 1, 2)
                plt.cla()
                plt.plot(np.linspace(0,self.filter_cnt, self.filter_cnt),
                         np.sqrt(np.sum((self.filt_vec[2:4,:]-self.filt_vec[5:7,:])**2,
                                        axis=0)), "-b")

                plt.grid(True)

                # Plot trace of cov matrix
                plt.subplot(3, 1, 3)
                plt.cla()
                plt.plot(np.linspace(0,self.filter_cnt, self.filter_cnt),
                         np.asarray(self.cov_traces), "-k")
                plt.grid(True)

            # Plot real pings vs expected meas
            if True:
                plt.subplot(1, 1, 1)
                plt.cla()
                plt.plot(self.pings_vec[:,1],
                         self.pings_vec[:,2], "-k")
                plt.plot(self.pings_vec[:,4],
                         self.pings_vec[:,5], "-b")

                # For debugging
                #  print (self.pings_vec[:, 2])
                #  print (self.pings_vec[:, 5])
                #  print (self.pings_vec[:, 2] - self.pings_vec[:, 5])
                #  print (np.linalg.norm(self.pings_vec[:, 2] - self.pings_vec[:, 5]))
                #  print(np.gradient(exp_mbes_ranges) - np.gradient(real_mbes_ranges))

                #  print(self.meas_cov)
                #  print (np.linalg.norm(exp_mbes_ranges - real_mbes_ranges))
                #  print (np.linalg.norm(np.gradient(real_mbes_ranges)
                #  - np.gradient(exp_mbes_ranges)))

                plt.grid(True)
            

            plt.pause(0.0001)

            if self.survey_finished:
                plt.savefig(self.survey_name+"_tracks.png")



if __name__ == "__main__":
    rospy.init_node("pf_statistics", disable_signals=False)
    try:
        PFStatsVisualization()
    except rospy.ROSInterruptException:
        pass
