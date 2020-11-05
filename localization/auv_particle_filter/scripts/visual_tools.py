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
from auv_particle import matrix_from_tf

class PFStatsVisualization(object):
    
    def __init__(self):
        stats_top = rospy.get_param('~pf_stats_top', 'stats')
        self.stats_sub = rospy.Subscriber(stats_top, numpy_msg(Floats), self.stat_cb)
        self.path_img = rospy.get_param('~background_img_path', 'default_real_mean_depth.png')
        self.map_frame = rospy.get_param('~map_frame', 'map')
        self.odom_frame = rospy.get_param('~odom_frame', 'odom')
        self.survey_name = rospy.get_param('~survey_name', 'survey')
        
        self.filter_cnt = 1
        self.filt_vec = np.zeros((14,1))
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

        rospy.spin()

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

        data_t = stat_msg.data.copy().reshape(14,1)
        # Rotate AUV trajectory to place wrt odom in the image
        data_t[2:5] = self.m2o_mat[0:3,0:3].dot(data_t[2:5])
        data_t[5:8] = self.m2o_mat[0:3,0:3].dot(data_t[5:8])
        
        # Reconstruct 3x3 covariance matrix
        # Not account for z values atm
        cov_mat = np.zeros((3,3))
        cov_mat[np.triu_indices(3, 0)] = np.asarray(data_t[8:14]).reshape(1,6)
        cov_mat[1,0] = cov_mat[0,1]
        cov_mat = (self.m2o_mat[0:3,0:3].transpose().dot(cov_mat)).dot(self.m2o_mat[0:3,0:3])
        data_t[8:14] = cov_mat[np.triu_indices(3)].reshape(6,1)

        self.filt_vec = np.hstack((self.filt_vec, data_t))
        self.filter_cnt += 1
        if self.filter_cnt > 0:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            # Plot N_eff
            #  plt.subplot(2, 1, 1)
            #  plt.cla()
            #  plt.plot(np.linspace(0,self.filter_cnt, self.filter_cnt),
                     #  self.filt_vec[0,:], "-k")
            #  plt.plot(np.linspace(0,self.filter_cnt, self.filter_cnt),
                     #  self.filt_vec[1,:], "-b")
           #
            #  plt.grid(True)
            
            # Plot x,y from odom and PF
            plt.cla()
            # Center image on odom frame
            plt.imshow(self.img, extent=[-647-self.m2o_mat[0,3], 1081-self.m2o_mat[0,3],
                                         -1190-self.m2o_mat[1,3], 523-self.m2o_mat[1,3]])
            #  plt.imshow(self.img, extent=[-740, 980, -690, 1023])
            plt.plot(self.filt_vec[2,:],
                     self.filt_vec[3,:], "-k")

            plt.plot(self.filt_vec[5,:],
                     self.filt_vec[6,:], "-r")

            self.plot_covariance_ellipse(self.filt_vec[5:7,-1], self.filt_vec[8:14,-1])

            plt.pause(0.001)

            if self.survey_finished:
                plt.savefig(self.survey_name+"_tracks.png")



if __name__ == "__main__":
    rospy.init_node("pf_statistics", disable_signals=False)
    try:
        PFStatsVisualization()
    except rospy.ROSInterruptException:
        pass
