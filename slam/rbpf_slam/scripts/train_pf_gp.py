#!/usr/bin/env python3

import rospy
import torch
from sensor_msgs.msg import PointCloud2
import matplotlib.pyplot as plt
from bathy_gps.gp import SVGP
import numpy as np
import os
from auvlib.data_tools import benchmark
from auvlib.data_tools import std_data, xyz_data
from scipy.spatial.transform import Rotation as rot

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
        
        self.data_path = '/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/data/results/'
        # _, _, self.names = next(os.walk(self.data_path + 'particles/'))
        self.pc = 1 #len(self.names)
        self.particle = np.empty(self.pc, dtype=object)
        self.n = 1000
        self.trainNplot()
        # print(names)


    def trainNplot(self):
        # train particles gps
        cloud = np.load(self.data_path + 'ping_cloud.npy')
        targets = cloud[:,2]
        for i in range (0,1):
            self.particle[i] = create_particle(self.n)
            # inputs = np.load(self.data_path + 'particles/' + self.names[i])
            inputs = cloud[:, :2]
            print(len(inputs))
            print(len(targets))

            self.particle[i].gp.fit(inputs, targets, n_samples=1000, max_iter=1000, learning_rate=1e-1, rtol=1e-4, ntol=100, auto=False, verbose=True)
            print('\nParticle {} done ({} in total) \n'.format(i, self.pc))
            self.particle[i].gp.plot(inputs, targets, self.data_path + 'training.png', n=100, n_contours=100 )
            gp_path = self.data_path + 'svgp.pth'
            self.particle[i].gp.save(gp_path)
            # save the posterior point cloud
            self.plot_and_save_posterior(inputs, targets, gp_path)

    def plot_and_save_posterior(self, inputs, targets, path):
        x = inputs[:,0]
        y = inputs[:,1]
        gp = SVGP.load(self.n, path )
        gp.plot(inputs, targets, self.data_path + 'posterior.png')
        gp.save_posterior(self.n, min(x), max(x), min(y), max(y), self.data_path + 'posterior.npy', verbose=False)


    def rms(self):
        gp_cloud = np.load(self.data_path + 'posterior.npy')
        ping_cloud = np.load(self.data_path + 'ping_cloud.npy')

        cloud = [ping_cloud, gp_cloud[:,0:3]]

        # Ground truth benchmark
        bm = benchmark.track_error_benchmark()
        bm.add_ground_truth(cloud, cloud)

        ping_list = [ping_cloud]
        bm.add_benchmark(ping_list, ping_list, "real")

        gp_list = [gp_cloud[:,0:3]]
        bm.add_benchmark(gp_list, gp_list, "gp")


        bm.print_summary()

class create_particle():
    def __init__(self, n):
        self.gp = SVGP(n) 
        # self.gp = SVGP.load(1000, gp_path)






if __name__ == '__main__':
    working = Train_gps()
    print('\nnow time for rms\n')
    working.rms()