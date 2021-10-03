#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import time
import numpy as np
import os

# --------------- learning the basics ----------------
def learn_basic():
    hej = np.array([[0, 1], [3, 4], [2,5],[0, 1], [3, 4], [2,5]])
    da = np.zeros((6,))
    # da = np.array([[1], [2], [3], [4], [5], [6]])
    print('shape of hej \n', hej.shape)
    print(hej)
    print('shape of da \n', da.shape)
    print(da)
    print(len(da))
    new = np.zeros((len(da),3))
    new[:,:2] = hej
    print(new)
    print('\n now add da \n')
    new[:,2] = da
    print(new)
    print('shape of new \n', new.shape)


    # da = np.append(da, da, axis=0)
    # print('shape of da \n', da.shape)
    # print(da)
    hej = np.insert(hej, da, axis=1)
    # hej = np.append(hej, [[da]], axis=0)
    print('shape of hej \n', hej.shape)
    print(hej)
# hej = 1
# da = 1
# new = np.empty((1,2))
# print(new.shape)

# new = np.append(new, [[hej,da]],axis=0)
# hej = 2
# da = 4
# new = np.append(new, [[hej,da]],axis=0)

# # new = np.append(da, axis=1)

# print(new.shape)
# print(new)
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

# class plotGPs():

#     def __init__(self):

def test_plot():
    #plot input before training
    root = '/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/slam/rbpf_slam/data/record_pf2train_gp/results/xy/'
    _, _, names = next(os.walk(root))
    print(names[0])
    for f in names: 
        p1 = np.load(root + f)
        print('particle xy pose', p1.shape)
        xx = p1[:,0]
        yy = p1[:,1]
        plt.plot(xx,yy, 'b')
        plt.xlabel('x axis (m)')
        plt.ylabel('y axis (m)')
        plt.show()

    root = '/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/slam/rbpf_slam/data/record_pf2train_gp/results/particles/'
    _, _, names = next(os.walk(root))
    print(names[0])
    for f in names: 
        p1 = np.load(root + f)
        print('input', p1.shape)
        xx = p1[:,0]
        yy = p1[:,1]
        plt.plot(xx,yy, 'b')
        plt.xlabel('x axis (m)')
        plt.ylabel('y axis (m)')
        plt.show()

def gp_data():    
    # plot saved poserior
    root = '/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/slam/rbpf_slam/data/particle_posterior/'
    _, _, names = next(os.walk(root))
    # print(names[0])
    for f in names:
        p1 = np.load(root + f)
        print('after regression', p1.shape)
        xx = p1[:,0]
        yy = p1[:,1]
        mu = p1[:,2]
        sigma = p1[:,3]
        # plot_fit(xx, yy, mu, sigma)
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

def plot_fit(x,y,mu,var, m_y='k-o', m_mu='b-<', l_y='true', l_mu='predicted', legend=True, title=''):
    """
    Plot the fit of a GP
    """
    if y is not None:
        plt.plot(x,y, m_y, label=l_y)
    plt.plot(x,mu, m_mu, label=l_mu)
    vv = 2*np.sqrt(var)
    plt.fill_between(x, (mu-vv), (mu+vv), alpha=0.2)#, edgecolor='gray', facecolor='cyan')
    # plt.fill_between(x[:,0], (mu-vv)[:,0], (mu+vv)[:,0], alpha=0.2, edgecolor='gray', facecolor='cyan')
    if legend:
        plt.legend()
    if title != '':
        plt.title(title)
    plt.show()

if __name__ == '__main__':
    # gp_data()
    learn_basic()
    # test_plot()