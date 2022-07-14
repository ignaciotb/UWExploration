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


def plot_rbpf_solution(avg_pos_out, avg_ors_out, pos_in, ors_in, data_in):

    # print(avg_pos_out.shape)
    # print(beams_0.shape)

    i = 0
    j = 0
    rot_mat_avg = avg_ors_out[i]
    rot_mat_in = ors_in[i]

    beams_in = data_in["beams"]
    # beams_out = data_out["beams"]   # for testing
    beams_tf = []
    for beam in beams_in:
        if j == 100:
            j = 0
            i += 1
            rot_mat_avg = avg_ors_out[i]
            rot_mat_in = ors_in[i]

        # Back to map origin
        beam = np.add(beam.T, -pos_in[i])
        beam = np.matmul(rot_mat_in.T, beam)

        # Transform to avg pose
        beam = np.matmul(rot_mat_avg, beam)
        beam = np.add(beam.T, avg_pos_out[i])
        beams_tf.append(beam)
        j += 1

    beams_tf = np.asarray(beams_tf).reshape((-1, 3))
    fig, ax = plt.subplots(1, sharex=True, sharey=True)
    cr = ax.scatter(beams_tf[:, 0], beams_tf[:, 1], c=beams_tf[:, 2],
                        cmap='jet', s=0.4, edgecolors='none')

    # for testing
    # cr = ax.scatter(beams_out[:, 0], beams_out[:, 1], c=beams_out[:, 2],
    #                     cmap='jet', s=0.4, edgecolors='none')
    
    fig.colorbar(cr, ax=ax)
    ax.plot(avg_pos_out[:,0], avg_pos_out[:,1], "-r", linewidth=0.2)
    ax.plot(pos_in[:,0], pos_in[:,1], "-b", linewidth=0.2)

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

    data_out = np.load(os.path.join(path, "data_particle_29.npz"))
    positions_all.append(data_out["track_position"])
    orientations_all.append(data_out["track_orientation"])
    
    positions = np.asarray(positions_all)
    positions = np.reshape(positions, (-1,particle_count*3)) 
    
    orientations = np.asarray(orientations_all)
    orientations = np.reshape(orientations, (orientations.shape[1], 3)) 
    orientations = np.reshape(orientations, (-1, 9)) 
    # print(orientations)

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

    avg_pos_out = np.asarray(avg_positions)
    avg_ors_out = np.asarray(avg_or)
    # print(avg_pos)
    # print(avg_ors)

    #### Load particle 0 data to be moved to average trajectory
    positions_in = []
    orientations_in = []
    data_in = np.load(os.path.join(path, "data_particle_0.npz"))
    positions_in.append(data_in["track_position"])
    orientations_in.append(data_in["track_orientation"])
    
    positions = np.asarray(positions_in)
    positions = np.reshape(positions, (-1, 3)) 
    orientations = np.asarray(orientations_in)
    orientations = np.reshape(orientations, (-1, 9)) 

    avg_or = []
    for ori in orientations:
        rot_mat_0 = ori.reshape(3,3).T
        avg_or.append(rot_mat_0)

    pos_in = positions
    ors_in = np.asarray(avg_or)
    print(pos_in.shape)
    print(ors_in.shape)


    plot_rbpf_solution(avg_pos_out, avg_ors_out, pos_in, ors_in, data_in)