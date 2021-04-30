#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import csv
import time
import numpy as np
import os

root = '/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/data/results/'

obs_path = root + 'obs_path/'
tr_path = root + 'tr_path/'

_, _, name_obs = next(os.walk(obs_path))
_, _, name_tr = next(os.walk(tr_path))

n = len(name_obs)
p_tr = [None]*n
p_obs = [None]*n

for i in range(n): 
    p_obs[i] = np.load(obs_path + name_obs[i])
    p_tr[i] = np.load(tr_path + name_tr[i])

    x_obs = p_obs[i][:,0]
    y_obs = p_obs[i][:,1]
    x_tr = p_tr[i][:,0]
    y_tr = p_tr[i][:,1]
    plt.plot(x_obs, y_obs , color = 'black')#, label = 'observation')
    plt.plot(x_tr, y_tr ) #, label = 'trajectory' )

# plt.legend()
plt.xlabel('x axis (m)')
plt.ylabel('y axis (m)')
plt.show()



