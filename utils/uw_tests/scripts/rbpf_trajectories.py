#!/usr/bin/env python3

from audioop import avg
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rot

def plot_errors(filt_vec, particle_count, filter_cnt, path):

    # 19998 For test 6
    # 30000 For test 3
    
    # cov_traces = []
    # for i in filt_vec:
    #     cov_mat = np.zeros((3,3))
    #     cov_mat[np.triu_indices(3, 0)] = np.asarray(i[11:17]).reshape(-1,6)
    #     cov_mat[1,0] = cov_mat[0,1]
    #     # cov_mat = (self.m2o_mat[0:3,0:3].transpose().dot(cov_mat)).dot(self.m2o_mat[0:3,0:3])
    #     i[11:17] = cov_mat[np.triu_indices(3)].reshape(6,1)
    #     cov_traces.append(np.trace(cov_mat))

    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.cla()
    t_steps = np.linspace(5,30000, 30000-5)

    # Error between GT and DR
    plt.plot(t_steps,
                np.linalg.norm(filt_vec[2:4,5:30000]-filt_vec[8:10,5:30000], axis=0), "-r")

    # Error between GT and filter
    plt.plot(t_steps,
                np.linalg.norm(filt_vec[2:4,5:30000]-filt_vec[5:7,5:30000], axis=0), "-b")

    plt.grid(True)

    # Plot N_eff against threshold
    plt.subplot(2, 1, 2)
    plt.cla()
    plt.plot(t_steps,
            np.tile(np.asarray(particle_count/2.), (30000-5, 1)), "-r")
    plt.plot(t_steps,
            np.asarray(filt_vec[0, 5:30000]), "-g")
    plt.grid(True)

    # Plot trace of cov matrix
    # plt.subplot(3, 1, 3)
    # plt.cla()
    # plt.plot(t_steps,
    #             np.asarray(cov_traces), "-k")
    # plt.grid(True)

    plt.savefig(path + "/errors_test.png")


def plot_trajectories(filt_vec, beams, path):
    
    # fig, ax = plt.subplots(1, sharex=True, sharey=True)

    # cr = ax.scatter(beams[:, 0], beams[:, 1], c=beams[:, 2],
    #                     cmap='jet', s=0.4, edgecolors='none')
    
    # fig.colorbar(cr, ax=ax)
    # ax.plot(filt_vec[2,1:], filt_vec[3,1:], "-k", linewidth=0.3)
    # ax.plot(filt_vec[5,1:], filt_vec[6,1:], "-b", linewidth=0.3)
    # ax.plot(filt_vec[8,1:], filt_vec[9,1:], "-r", linewidth=0.3)

    # fig.savefig(path + "/rbpf_trajectories.png", bbox_inches='tight', dpi=1000)

    # RMSE
    # GT and DR
    print("RMSE GT and DR")
    norms = np.linalg.norm(filt_vec[2:4,5:30000]-filt_vec[8:10,5:30000], axis=0)
    print(np.sum(norms)/len(norms))
    # GT and filter
    print("RMSE GT and RBPF")
    norms = np.linalg.norm(filt_vec[2:4,5:30000]-filt_vec[5:7,5:30000], axis=0)
    print(np.sum(norms)/len(norms))


if __name__ == '__main__':

    path = sys.argv[1]  # /media/orin/Seagate\ Expansion\ Drive/rbpf_results/lolo_0/  
    experiment = str(sys.argv[2]) # Experiment number

    filt_vec = np.load(path + '/' + experiment + '/trajectories.npz')
    filt_vec = filt_vec['full_dataset']

    # beams = np.load(path + '/map_mbes.npy')
    # plot_trajectories(filt_vec, beams, path + '/' + experiment)
    plot_trajectories(filt_vec, None, path + '/' + experiment)

    plot_errors(filt_vec, 100, filt_vec.shape[1], path + '/' + experiment)
