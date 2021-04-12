#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import time
import numpy as np
import os

# --------------- learning the basics ----------------
# hej = np.array([[0, 1], [3, 4], [2,5]])
# da = np.array([[6, 6],[7,7], [8,8]])
# print(hej.shape)
# print(da)
# # np.append(hej[:,0], da[:,0], axis=0)
# # np.append(hej[:,0], da[:,1], axis=1)
# hej = np.append(hej, da, axis=0)

# print(hej.shape)
# print(hej)

# root = '/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/data/'
# cloud = np.load(root+'pcl_33_over.npy')
# print(cloud.shape) # (2391787, 3)
#plot input before training
root = '/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/data/record_pf2train_gp/results/particles/'
_, _, names = next(os.walk(root))
print(names[0])
for f in names: 
    p1 = np.load(root + f)
    # print(p1.shape)
    xx = p1[:,0]
    yy = p1[:,1]
    plt.plot(xx,yy, 'b')
    plt.xlabel('x axis (m)')
    plt.ylabel('y axis (m)')
    plt.show()

# plot saved poserior
root = '/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/data/particle_posterior/'
_, _, names = next(os.walk(root))
# print(names[0])
for f in names:
    p1 = np.load(root + f)
    # print(p1.shape)
    xx = p1[:,0]
    yy = p1[:,1]
    mu = p1[:,2]
    sigma = p1[:,3]
    fig = plt.figure(figsize=(6,6), constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    ax1 = fig.add_subplot(gs[:2, :])
    plt.plot(xx,yy, 'b')
    plt.xlabel('x axis (m)')
    plt.ylabel('y axis (m)')
    ax2 = fig.add_subplot(gs[2, :])
    plt.plot(mu,sigma, 'g')
    plt.xlabel('mu')
    plt.ylabel('sigma')
    plt.show()
