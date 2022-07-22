#!/usr/bin/env python3

from audioop import avg
import os
import shutil
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as rot


# plot_rbpf_solution(avg_pos_out, avg_ors_out, data_out, avg_pos_in, avg_ors_in, data_in)


def plot_rbpf_solution(avg_pos_out, avg_ors_out, pos_in, ors_in, pos_particles, data_in, path):

    i = 0
    j = 0
    rot_mat_avg = avg_ors_out[i]
    rot_mat_in = ors_in[i]

    beams_in = data_in["beams"]
    beams_tf = []
    for beam in beams_in:
        if j == 100:    # Number of beams per ping used in survey
            j = 0
            i += 1
            rot_mat_avg = avg_ors_out[i]
            rot_mat_in = ors_in[i]

        # Back to map origin
        beam = np.add(beam.T, -pos_in[i])
        beam = np.matmul(rot_mat_in.T, beam)

        # Transform to avg pose
        beam = np.matmul(rot_mat_avg, beam.T)
        beam = np.add(beam.T, avg_pos_out[i])
        beams_tf.append(beam)
        j += 1

    beams_tf = np.asarray(beams_tf).reshape((-1, 3))
    
    fig, ax = plt.subplots(1, sharex=True, sharey=True)

    # # For testing
    # beams_in = data_in["beams"]
    # cr = ax.scatter(beams_in[:, 0], beams_in[:, 1], c=beams_in[:, 2],
    #                     cmap='jet', s=0.4, edgecolors='none')

    cr = ax.scatter(beams_tf[:, 0], beams_tf[:, 1], c=beams_tf[:, 2],
                        cmap='jet', s=0.4, edgecolors='none')
    
    fig.colorbar(cr, ax=ax)
    ax.plot(avg_pos_out[:,0], avg_pos_out[:,1], "-b", linewidth=0.2)
    
    ax.plot(pos_particles[:,0], pos_particles[:,1], "-r", linewidth=0.2)

    fig.savefig(path + "/rbpf_map.png", bbox_inches='tight', dpi=1000)


if __name__ == '__main__':

    path = sys.argv[1]    
    particle_count = int(sys.argv[2])

    positions_all = []
    orientations_all = []
    i = 0
    for file in os.listdir(path):
        if file.endswith(".npz") and not file.startswith("trajectories") and i < particle_count:
            print(os.path.join(path, file))
            data = np.load(os.path.join(path, file))
            positions_all.append(data["track_position"])
            orientations_all.append(data["track_orientation"])
            i += 1
   
    positions = np.asarray(positions_all)
    orientations = np.asarray(orientations_all)
    positions_arranged = positions[0]
    orientations_arranged = orientations[0]

    for i in range(1, positions.shape[0]):
        positions_arranged = np.hstack((positions_arranged, positions[i]))
        orientations_arranged = np.hstack((orientations_arranged, orientations[i]))
    
    # Eigen decomposition of Q*Qt for quaternions
    avg_positions = []
    avg_or = []
    for row in range(0, positions_arranged.shape[0]):
    # for row in range(0, 1):
        # Positions
        step_pos = positions_arranged[row, :].reshape(-1, 3).sum(0)/particle_count
        avg_positions.append(step_pos)

        # Orientations: need to be converted to quaternions for averaging
        ors_col = orientations_arranged[row, :].reshape(-1, 3)
        ors_col_quats = np.zeros((1,4))
        for row_i in ors_col:
            ors_col_quats = np.vstack((ors_col_quats, rot.from_euler("xyz", row_i, degrees=False).as_quat()))

        if True:
            ors_col_quats_new = ors_col_quats[1:,:]
            # Eigen decomposition of matrix of quats
            w, v = np.linalg.eigh(np.matmul(ors_col_quats_new.T, ors_col_quats_new))
            idx_max = np.where(w == w.max())
            avg_or.append(rot.from_quat(v.T[3]).as_matrix())
        else:
            # Left here for the memory of it
            ors_col_quats = ors_col_quats.sum(0)/particle_count
            avg_or.append(rot.from_quat(ors_col_quats).as_matrix())

    avg_pos = np.asarray(avg_positions)
    avg_ors = np.asarray(avg_or)

    #### Use one single particle's data to be transformed to computed average trajectory
    positions_in = []
    orientations_in = []
    positions_in.append(data["track_position"])
    orientations_in.append(data["track_orientation"])
    
    positions = np.asarray(positions_in)
    positions = np.reshape(positions, (-1, 3)) 
    orientations = np.asarray(orientations_in)
    orientations = np.reshape(orientations, (-1, 3)) 

    avg_or = []
    for ori in orientations:
        rot_avg = rot.from_euler("xyz", ori, degrees=False).as_matrix()
        avg_or.append(rot_avg)

    pos_in = positions
    ors_in = np.asarray(avg_or)

    plot_rbpf_solution(avg_pos, avg_ors, pos_in, ors_in, positions_arranged, data, path)