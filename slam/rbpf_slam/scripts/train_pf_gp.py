#!/usr/bin/env python3

import rospy
import torch
from sensor_msgs.msg import PointCloud2
import matplotlib.pyplot as plt
from bathy_gps.gp import SVGP
import numpy as np
import os

# train in a new node to save time 
class Train_gps():

    def __init__(self):
        # --------- old --------
        # Create publisher
        # self.pub = rospy.Publisher('/trained_gps', PointCloud2, queue_size=1)
        # Subscribe to particles
        # rospy.Subscriber('/gps2train', PointCloud2, self.cb)
        # root = '/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/data/'
        # self.gp_path = root + 'svgp.pth'
        # ----------------------
        
        self.data_path = '/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/data/record_pf2train_gp/results/'
        _, _, self.names = next(os.walk(self.data_path + 'particles/'))
        self.pc = len(self.names)
        self.particle = np.empty(self.pc, dtype=object)
        self.trainNplot()
        # print(names)


    def trainNplot(self):
        # train particles gps
        targets = np.load(self.data_path + 'target_gp.npy')
        for i in range (0,1):
            self.particle[i] = create_particle()
            inputs = np.load(self.data_path + 'particles/' + self.names[i])
            print(len(inputs))
            print(len(targets))

            self.particle[i].gp.fit(inputs, targets, n_samples=100, max_iter=100, learning_rate=1e-1, rtol=1e-4, ntol=100, auto=False, verbose=True)
            print('\nParticle {} done ({} in total) \n'.format(i, self.pc))
            self.particle[i].gp.plot(inputs, targets, self.data_path + 'training.png', n=100, n_contours=100 )
            gp_path = self.data_path + 'svgp.pth'
            self.particle[i].gp.save(gp_path)
            # save the posterior point cloud
            self.plot_and_save_posterior(inputs, targets, gp_path)
            
            # fig, ax = plt_settings()
            # Regression_result(fig, ax)
            # self.particle[i].gp.save_posterior(100, min(x), max(x), min(y), max(y),'../data/particle_posterior/particle'+str(i)+'.npy', verbose=False)

    def plot_and_save_posterior(self, inputs, targets, path):
        x = inputs[:,0]
        y = inputs[:,1]
        gp = SVGP.load(100, path )
        gp.plot(inputs, targets, self.data_path + 'posterior.png')
        gp.save_posterior(100, min(x), max(x), min(y), max(y), self.data_path + 'posterior.npy', verbose=False)

class create_particle():
    def __init__(self):
        self.gp = SVGP(100) 
        # self.gp = SVGP.load(1000, gp_path)






if __name__ == '__main__':
    Train_gps()