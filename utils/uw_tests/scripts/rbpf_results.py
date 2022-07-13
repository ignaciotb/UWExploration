#!/usr/bin/env python3

from audioop import avg
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rot


def plot_rbpf_solution(avg_pos_0, avg_or_0, avg_pos_29, avg_or_29, data_0, data_29):

    # print(avg_pos_0.shape)
    # print(beams_0.shape)

    i = 0
    j = 0
    rot_mat_0 = avg_or_0[i]
    rot_mat_29 = avg_or_29[i]

    beams_29 = data_29["beams"]
    beams_0 = data_0["beams"]
    beams_tf = []
    for beam in beams_29:
        if j == 100:
            j = 0
            i += 1
            rot_mat_0 = avg_or_0[i]
            rot_mat_29 = avg_or_29[i]

        # Back to map origin
        beam = np.add(beam.T, -avg_pos_29[i])
        beam = np.matmul(rot_mat_29.T, beam)

        # Transform to avg pose
        beam = np.matmul(rot_mat_0, beam)
        beam = np.add(beam.T, avg_pos_0[i])
        beams_tf.append(beam)
        j += 1

    beams_tf = np.asarray(beams_tf).reshape((-1, 3))
    fig, ax = plt.subplots(1, sharex=True, sharey=True)
    # cr = ax.scatter(beams_tf[:, 0], beams_tf[:, 1], c=beams_tf[:, 2],
    #                     cmap='jet', s=0.4, edgecolors='none')

    cr = ax.scatter(beams_0[:, 0], beams_0[:, 1], c=beams_0[:, 2],
                        cmap='jet', s=0.4, edgecolors='none')
    fig.colorbar(cr, ax=ax)

    ax.plot(avg_pos_0[:,0], avg_pos_0[:,1], "-r", linewidth=0.2)
    ax.plot(avg_pos_29[:,0], avg_pos_29[:,1], "-b", linewidth=0.2)

    fig.savefig("./rbpf_map.png", bbox_inches='tight', dpi=1000)


if __name__ == '__main__':

    i = sys.argv[1]    
    path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'rbpf/lolo_0/' + str(i)))
    particle_count = 1

    # Path(results_dir).mkdir(parents=True, exist_ok=False)
    positions_all = []
    orientations_all = []
    # for file in os.listdir(path):
    #     if file.endswith(".npz") and not file.startswith("gt"):
    #         # print(os.path.join(path, file))
    #         data = np.load(os.path.join(path, file))
    #         positions_all.append(data["track_position"])
    #         print(data["track_position"].shape)
    #         orientations_all.append(data["track_orientation"])

    data_0 = np.load(os.path.join(path, "data_particle_0.npz"))
    positions_all.append(data_0["track_position"])
    orientations_all.append(data_0["track_orientation"])
    
    positions = np.asarray(positions_all)
    positions = np.reshape(positions, (-1,particle_count*3)) 
    
    orientations = np.asarray(orientations_all)
    orientations = np.reshape(orientations, (orientations.shape[1], 3)) 
    orientations = np.reshape(orientations, (-1, 9)) 
    print(orientations)

    avg_or = []
    for ori in orientations:
        rot_mat_0 = ori.reshape(3,3).T
        avg_or.append(rot_mat_0)
        # print(rot_mat_0)



    # # Average poses of particles
    avg_positions = []
    # avg_or = []
    for row in range(0, positions.shape[0]):
        step_pos = positions[row, :].reshape(-1, 3).sum(0)/particle_count
        avg_positions.append(step_pos)

    #     step_or = orientations[row, :].reshape(-1, 3).sum(0)/particle_count
    #     avg_or.append(step_or)

    avg_pos_0 = np.asarray(avg_positions)
    avg_ors_0 = np.asarray(avg_or)
    # print(avg_pos)
    # print(avg_ors)

    # # Load particle 0 data
    positions_all = []
    orientations_all = []
    data_29 = np.load(os.path.join(path, "data_particle_29.npz"))
    positions_all.append(data_29["track_position"])
    orientations_all.append(data_29["track_orientation"])
    
    positions = np.asarray(positions_all)
    positions = np.reshape(positions, (-1,particle_count*3)) 
    
    orientations = np.asarray(orientations_all)
    orientations = np.reshape(orientations, (orientations.shape[1], 3)) 
    orientations = np.reshape(orientations, (-1, 9)) 
    print(orientations)

    avg_or = []
    for ori in orientations:
        rot_mat_0 = ori.reshape(3,3).T
        avg_or.append(rot_mat_0)
        # print(rot_mat_0)

    # # Average poses of particles
    avg_positions = []
    # avg_or = []
    for row in range(0, positions.shape[0]):
        step_pos = positions[row, :].reshape(-1, 3).sum(0)/particle_count
        avg_positions.append(step_pos)

    #     step_or = orientations[row, :].reshape(-1, 3).sum(0)/particle_count
    #     avg_or.append(step_or)

    avg_pos_29 = np.asarray(avg_positions)
    avg_ors_29 = np.asarray(avg_or)

    plot_rbpf_solution(avg_pos_0, avg_ors_0, avg_pos_29, avg_ors_29, data_0, data_29)