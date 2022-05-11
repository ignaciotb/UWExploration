#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import os

from scipy.spatial.transform import Rotation as Rot
from rospy.numpy_msg import numpy_msg
from sensor_msgs.msg import PointCloud2
from sklearn.metrics import mean_squared_error

class pf_stats(object):

    def __init__(self, survey_name):

        self.survey_name = survey_name
        dataset = np.load(survey_name)
        self.filt_vec = dataset["full_dataset"]
        self.filter_cnt = self.filt_vec.shape[1]
        self.cov_traces = [0.]

    def compute_errors(self):

        results = []
        rms_dr = mean_squared_error(self.filt_vec[2:5,:], self.filt_vec[10:13,:], squared=False)
        rms_pf = mean_squared_error(self.filt_vec[2:5,:], self.filt_vec[6:9,:], squared=False)
        results.append(rms_dr)
        results.append(rms_pf)

        # Wrap yaw between -pi,pi
        self.filt_vec[5,:] = [(a + np.pi) % (2 * np.pi) - np.pi for a in self.filt_vec[5,:]]
        self.filt_vec[9,:] = [(a + np.pi) % (2 * np.pi) - np.pi for a in self.filt_vec[9,:]]
        self.filt_vec[13,:] = [(a + np.pi) % (2 * np.pi) - np.pi for a in self.filt_vec[13,:]]

        rms_yaw_pf = mean_squared_error(self.filt_vec[5,:], self.filt_vec[9,:], squared=False)
        rms_yaw_dr = mean_squared_error(self.filt_vec[5,:], self.filt_vec[13,:], squared=False)
        results.append(rms_yaw_dr)
        results.append(rms_yaw_pf)

        # print("RMSE DR ", rms_dr)
        # print("RMSE PF ", rms_pf)
        # print("RMSE yaw DR ", rms_yaw_dr)
        # print("RMSE yaw PF ", rms_yaw_pf)

        d_opt = 0.
        for i in range(1, self.filt_vec.shape[1]):
            cov_mat = np.zeros((3,3))
            cov_mat[np.triu_indices(3, 0)] = np.asarray(self.filt_vec[14:20,i]).reshape(1,6)
            cov_mat[1,0] = cov_mat[0,1]
            cov_mat[2,2] = 0.0001
            d_opt += np.exp(np.log(np.linalg.det(cov_mat)**(1/3.)))

        d_opt /= float(self.filt_vec.shape[1])
        # print("D-opt PF ", d_opt)
        results.append(d_opt)

        return results
        

    def plot_save(self, bg_img):
            
        # plt.gcf().canvas.mpl_connect('key_release_event',
        #         lambda event: [exit(0) if event.key == 'escape' else None])

        # Plot x,y from GT, odom and PF
        if True:
            plt.cla()
            #  Center image on odom frame
            #self.m2o_mat = np.array([-2.96770302e-01, -9.54948893e-01,  0.00000000e+00,
            #2.58061615e+02, 9.54948893e-01, -2.96770302e-01,  0.00000000e+00,
            #-7.28957764e+02, 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,
            #-5.98172493e+01, 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
            #1.00000000e+00])

            self.m2o_mat = identity matrix (4,4)
            self.m2o_mat = self.m2o_mat.reshape([4,4])


            for i in range(0, self.filt_vec.shape[1]):
                self.filt_vec[2:5, i] = self.m2o_mat[0:3,0:3].dot(self.filt_vec[2:5, i])
                self.filt_vec[6:9, i] = self.m2o_mat[0:3,0:3].dot(self.filt_vec[6:9, i])
                self.filt_vec[10:13, i] = self.m2o_mat[0:3,0:3].dot(self.filt_vec[10:13, i])

                cov_mat = np.zeros((3,3))
                cov_mat[np.triu_indices(3, 0)] = np.asarray(self.filt_vec[14:20, i]).reshape(1,6)
                cov_mat[1,0] = cov_mat[0,1]
                cov_mat = (self.m2o_mat[0:3,0:3].transpose().dot(cov_mat)).dot(self.m2o_mat[0:3,0:3])
                self.filt_vec[14:20, i] = cov_mat[np.triu_indices(3)].reshape(6)
                self.cov_traces.append(np.trace(cov_mat))

            plt.imshow(bg_img, extent=[-647-self.m2o_mat[0,3], 1081-self.m2o_mat[0,3],
                                            -1190-self.m2o_mat[1,3], 523-self.m2o_mat[1,3]])
            # plt.imshow(bg_img, extent=[-740, 980, -690, 1023])
            
            plt.xlim(100, -350)
            plt.ylim(600, -100)
            plt.axis('off')

            plt.plot(self.filt_vec[2,1:],
                        self.filt_vec[3,1:], "-k")
            plt.plot(self.filt_vec[10,1:],
                        self.filt_vec[11,1:], "-r")
            plt.plot(self.filt_vec[6,1:],
                        self.filt_vec[7,1:], "-g")

            for i in np.linspace(0,self.filt_vec.shape[1]-1,10):
                i = int(i)
                self.plot_covariance_ellipse(self.filt_vec[6:8,i], self.filt_vec[14:20,i])
            plt.savefig(self.survey_name + "_traj.png")

        # Plot error between DR PF and GT
        if False:
            plt.subplot(2, 1, 1)
            plt.cla()
            plt.plot(np.linspace(0,self.filter_cnt, self.filter_cnt),
                        np.sqrt(np.sum((self.filt_vec[2:4,:]-self.filt_vec[10:12,:])**2,
                                    axis=0)), "-k")
            plt.grid(True)

            # Error between PF and GT
            plt.subplot(2, 1, 2)
            plt.cla()
            plt.plot(np.linspace(0,self.filter_cnt, self.filter_cnt),
                        np.sqrt(np.sum((self.filt_vec[2:4,:]-self.filt_vec[6:8,:])**2,
                                    axis=0)), "-b")

            plt.grid(True)

            # Plot trace of cov matrix
            # plt.subplot(3, 1, 3)
            # plt.cla()
            # plt.plot(np.linspace(0,self.filter_cnt, self.filter_cnt),
            #          np.asarray(self.cov_traces), "-k")

            # Plot N_eff
            # plt.plot(np.linspace(0, self.filter_cnt, self.filter_cnt),
            #         np.tile(np.asarray(self.particle_count/2.), (self.filter_cnt, 1)), "-r")
            # plt.plot(np.linspace(0, self.filter_cnt, self.filter_cnt),
            #         np.asarray(self.filt_vec[0, :]), "-k")
            plt.grid(True)
            plt.savefig(self.survey_name + "_errors.png")

        # Plot real pings vs expected meas
        if False:
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
            plt.savefig(survey_name + "_pings.png")
            # plt.grid(True)
        
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
        rot = Rot.from_euler('z', angle).as_matrix()[0:2, 0:2]
        fx = np.stack([x, y]).T @ rot

        px = np.array(fx[:, 0] + xEst[0]).flatten()
        py = np.array(fx[:, 1] + xEst[1]).flatten()
        plt.plot(px, py, "--b")


if __name__ == "__main__":
    path = sys.argv[1]
    mission_type = sys.argv[2]

    # bg_img = plt.imread(sys.argv[2])
    # pf_stats_obj.plot_save(bg_img)

    results = []
    for file in os.listdir(path):
        if file.startswith(mission_type) and file.endswith(".npz"):
            print("Mission ", path + "/" + file)
            pf_stats_obj = pf_stats(path + "/" + file)
            results.append(pf_stats_obj.compute_errors())

    avgs = np.sum(np.asarray(results), axis=0)/np.asarray(results).shape[0]

    print("Averages ", np.asarray(results).shape[0])
    print("RMSE DR ", avgs[0])
    print("RMSE PF ", avgs[1])
    print("RMSE yaw DR ", avgs[2])
    print("RMSE yaw PF ", avgs[3])
    print("D-opt PF ", avgs[4])

