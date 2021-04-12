#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from bathy_gps import gp
import numpy as np
import os

# train in a new node to save time 
class Train_gps():

    def __init__(self):

        # Create publisher
        # self.pub = rospy.Publisher('/trained_gps', PointCloud2, queue_size=1)
        # Subscribe to particles
        # rospy.Subscriber('/gps2train', PointCloud2, self.cb)
        root = '/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/data/'
        gp_path = root + 'svgp.pth'
        my_path = root + 'record_pf2train_gp/results_2021_4_10___16_1_34/particles/'
        _, _, names = next(os.walk(my_path))
        self.pc = len(names)
        self.particle = np.empty(self.pc, dtype=object)
        # print(names)
        targets = np.load(root + 'record_pf2train_gp/results_2021_4_10___16_1_34/target_gp.npy')
        for i in range (0,self.pc):
            self.particle[i] = trained_particle(gp_path)
            inputs = np.load(my_path + names[i])
            self.particle[i].gp.fit(inputs, targets, n_samples=6000, max_iter=1000, learning_rate=1e-1, rtol=1e-4, ntol=100, auto=False, verbose=True)
            print('\nParticle {} klar \n'.format(i))

    # def training(self):
    #     # train particles gps
    #     gp.fit(inputs, targets, n_samples=6000, max_iter=1000, learning_rate=1e-1, rtol=1e-4, ntol=100, auto=False, verbose=True)

class trained_particle():
    def __init__(self, gp_path):
        self.gp = gp.SVGP.load(1000, gp_path)

if __name__ == '__main__':
    Train_gps()