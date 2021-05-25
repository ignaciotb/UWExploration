#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
from matplotlib.cm import get_cmap
import csv
import time
import numpy as np
import os

root = '/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/data/results/trajectory6/'
_, dir_name, _ = next(os.walk(root)) 
# dir_name.remove('gp_plot')

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
        name = 'Pastel1'  # 'Set3' #'tab20c' # 'Pastel1' #"Accent" 'tab20'
        cmap = get_cmap(name)  # type: matplotlib.colors.ListedColormap
        col = cmap.colors  # type: list
        

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
    name = 'Trajectory with 50 particles, 25 beam per ping'
    plt.title(name)
    plt.show()

def mapping(plot_type, Xd):
    fig = plt.figure()
    if Xd == '3d':
        # ax1 = fig.axes(projection='3d')
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    else:
        ax1 = fig.add_subplot(1, 2, 1)#, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2)#, projection='3d')

    for p in dir_name:
        map_path = root + p  + '/mapping/'
        _, _, name_est = next(os.walk(map_path + 'est_map/'))
        _, _, name_real = next(os.walk(map_path + 'real_map/'))

        n = len(name_est)
        
        for i in range(n): 
            pm = np.load(map_path + 'est_map/' + name_est[i])
            X = pm[:,0]
            Y = pm[:,1]
            Z = pm[:,2]
            rm = np.load(map_path + 'real_map/' + name_real[i])
            x = rm[:,0]
            y = rm[:,1]
            z = rm[:,2]

            # debug
            # print('exp ', Y)
            # print('real ', y)

            if plot_type == 'scatter':
                ax1.scatter(X, Y, c=Z, cmap='viridis', linewidth=0.5)
                ax2.scatter(x, y, c=z, cmap='viridis', linewidth=0.5)
            elif plot_type == 'surf': # only with 3d
                ax1.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')
                ax2.plot_trisurf(x, y, z, cmap='viridis', edgecolor='none')
            elif plot_type == '2dZ': # z must be 2-dimensional
                ax1.plot_surface(X, Y, Z, rstride=1, cstride=1,
                                cmap='viridis', edgecolor='none')
                ax2.plot_surface(x, y, z, rstride=1, cstride=1,
                                cmap='viridis', edgecolor='none')
            # plt.show()

        # xx, yy = np.meshgrid(x_tr, y_tr, sparse=True)
        # z = np.sin(xx**2 + yy**2) / (xx**2 + yy**2)
        # h = plt.contourf(x_tr,y_tr,z)
    ax1.set_title('RBPF map')
    ax2.set_title('Real map')
    ax1.set_xlabel('x axis (m)')
    ax1.set_ylabel('y axis (m)')
    # ax1.set_zlabel('z axis (m)')
    # ax2.set_zlabel('z axis (m)')
    ax2.set_xlabel('x axis (m)')
    ax2.set_ylabel('y axis (m)')
    plt.show()

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
    p_path()
    mapping('scatter', '2d')