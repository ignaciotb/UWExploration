#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
from matplotlib.cm import get_cmap
from statistics import mean
import csv
import time
import numpy as np
import os

root = '/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/slam/rbpf_slam/data/results/trajectory/' # Need to update to the path where the data is saved
_, dir_name, _ = next(os.walk(root)) 
color_name = 'Pastel1'  # 'Set3' #'tab20c' # 'Pastel1' #"Accent" 'tab20'
cmap = get_cmap(color_name)  # type: matplotlib.colors.ListedColormap
col = cmap.colors  # type: list

def p_path():
    k = 0
    first_obs = True # to get legend
    for p in dir_name:
        obs_path = root + p + '/localization/obs_path/'
        tr_path = root + p + '/localization/tr_path/'
        _, _, name_obs = next(os.walk(obs_path))
        _, _, name_tr = next(os.walk(tr_path))
        n = len(name_obs)
        # col = iter(cm.rainbow(np.linspace(0,1,n)))
        
        for i in range(n): 
            p_obs = np.load(obs_path + name_obs[i])
            p_tr = np.load(tr_path + name_tr[i])

            if k >= len(col): # for coloring
                k=0

            x_obs = p_obs[:,0]
            y_obs = p_obs[:,1]
            x_tr = p_tr[:,0]
            y_tr = p_tr[:,1]

            if first_obs: # to get legend
                plt.plot(x_obs, y_obs , color = 'black', label = 'observation')
                first_obs = False
            else:
                plt.plot(x_obs, y_obs , color = 'black')
            plt.plot(x_tr, y_tr, color = col[k] ) #, label = 'trajectory' )
        k += 1

    plt.legend()
    plt.xlabel('x axis (m)')
    plt.ylabel('y axis (m)')
    title_name = 'Trajectory with 50 particles, 25 beam per ping'
    plt.title(title_name)
    plt.show()



def Mapping(plot_type, Xd): # using this function to find the best particle and ancestors
    
    n_particle = len(dir_name)
    all_err = [0]*n_particle
    all_maps = [0]*n_particle
    n_data = 1.1e5

    # Upload data
    for j, p in enumerate(dir_name):
        map_path = root + p  + '/mapping/'
        _, _, name_est = next(os.walk(map_path + 'est_map/'))
        _, _, name_real = next(os.walk(map_path + 'obs_depth/'))
        n = len(name_est)
        mapping = np.zeros((1,3))
        err_vec = np.zeros((1,))
        for i in range(n): 
            pm = np.load(map_path + 'est_map/' + name_est[i])
            mapping = np.append(mapping, pm, axis=0)
            Z_est = pm[:,2]
            z_obs = np.load(map_path + 'obs_depth/' + name_real[i])
            try:
                err_vec = np.append(err_vec, abs(Z_est-z_obs))
            except:
                end_idx = len(Z_est)
                err_vec = np.append(err_vec, abs(Z_est-z_obs[0:end_idx]))

        all_err[j] = err_vec[1:]
        all_maps[j] = mapping[1:,:]
        if len(all_err[j]) < n_data:
            n_data = len(all_err[j])
    
    # Find the lowest error along the trajectory
    idx = find_best_trajectory(all_err, n_data)
    # Make error smooth
    depth_err = make_error_smooth(all_err[idx])
    # print('best trajectory: ', idx)
    print('best particle: ' + dir_name[idx])
    final_map = all_maps[idx]
    np.save(root + 'final_map.npy', final_map)
    X = final_map[:,0]
    Y = final_map[:,1]
    Z = final_map[:,2]

    # Plot the final map and error along the track
    fig = plt.figure()
    if plot_type == 'scatter' and Xd == '3d':
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.scatter(X, Y, Z, c=Z, cmap='viridis', linewidth=0.5)
        ax1.set_zlabel('z axis (m)')
    elif plot_type == 'scatter':
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(X, Y, c=Z, cmap='viridis', linewidth=0.5)
    elif plot_type == 'surf' and Xd == '3d': # only with 3d
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax1.set_zlabel('z axis (m)')
        ax1.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
    else:
        print('surf only works with 3D')
    # elif plot_type == '2dZ': # z must be 2-dimensional
    #     ax1.plot_surface(X, Y, Z_est, rstride=1, cstride=1,
    #                     cmap='viridis', edgecolor='none')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(depth_err)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Depth (m)')
    ax1.set_title('RBPF map ')
    ax2.set_title('Error')
    ax1.set_xlabel('x axis (m)')
    ax1.set_ylabel('y axis (m)')
    # plt.show()

def make_error_smooth(List):
    k = 25
    new_list = []
    for i in range(0,len(List), k):
        val = mean(List[i:i+k])
        new_list.append(val)
    return new_list

def find_best_trajectory(List, n_iter):
    idx = 0
    best_trajectory = []
    for ii in range(n_iter):
        val = 100.
        for jj in range(len(List)):
            if List[jj][ii] < val:
                val = List[jj][ii] 
                idx = jj
        best_trajectory.append(idx)
    return most_frequent(best_trajectory)

def most_frequent(List):
    return max(set(List), key = List.count)

# def err():
    # err = []
    # old_time_lap = 0
    # dist = [None]*n

    # time_lap = int(p_tr[-1])
    # p_tr = np.delete(p_tr, -1)

    # while len(x_obs) > len(x_tr):
        #     # print('x obs ', len(x_obs))
        #     # print('x tr ', len(x_tr))
        #     x_obs = np.delete(x_obs, 0)
        #     y_obs = np.delete(y_obs, 0)
        
        # while len(x_obs) < len(x_tr):
        #     # print('x obs ', len(x_obs))
        #     # print('x tr ', len(x_tr))
        #     x_tr = np.delete(x_tr, 0)
        #     y_tr = np.delete(y_tr, 0)
        
        # if old_time_lap < time_lap:
        #     # update the list with the mean errors
        #     print('hej')
        # old_time_lap = time_lap
        # # plot error
        # dist = np.sqrt((x_tr - x_obs)**2 + (y_tr - y_obs)**2)
        # print(dist)
        # plt.plot(dist)
        # plt.show()

    # MSE = np.mean(dist[:,0])
    # dist_arr = np.asarray(dist)
    # dist_mean = dist_arr.mean(axis=0)
    # print('mean ', dist_mean)
    # # for i in 
    # print(dist_arr.shape)
    # plt.plot(dist_arr)


if __name__ == '__main__':

    # p_path()
    plot_type = 'surf' # Chose from 'scatter' and 'surf'
    xd = '3d'             # Chose from '2d' and '3d'
    Mapping(plot_type, xd) 

    # result = np.load("/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/results/result.npz")
    # result_array = result['full_dataset']
    # print(result_array.shape)