#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
from bathy_gps import gp
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
        # ----------------------
        root = '/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/data/'
        self.gp_path = root + 'svgp.pth'
        self.data_path = root + 'record_pf2train_gp/results/'
        _, _, self.names = next(os.walk(self.data_path + 'particles/'))
        self.pc = len(self.names)
        self.particle = np.empty(self.pc, dtype=object)
        self.trainNplot()
        # print(names)


    def trainNplot(self):
        # train particles gps
        targets = np.load(self.data_path + 'target_gp.npy')
        for i in range (0,self.pc):
            self.particle[i] = create_particle(self.gp_path)
            inputs = np.load(self.data_path + 'particles/' + self.names[i])
            self.particle[i].gp.fit(inputs, targets, n_samples=6000, max_iter=1000, learning_rate=1e-1, rtol=1e-4, ntol=100, auto=False, verbose=True)
            print('\nParticle {} done ({} in total) \n'.format(i, self.pc))
            self.particle[i].gp.plot(inputs, targets,'../images/particle'+str(i)+'.png' )
            self.particle[i].gp.save('../data/particle_gp_model/particle'+str(i)+'.pth')
            # save the posterior point cloud
            x = inputs[:,0]
            y = inputs[:,1]
            self.particle[i].gp.save_posterior(1000, min(x), max(x), min(y), max(y),'../data/particle_posterior/particle'+str(i)+'.npy', verbose=False)

class create_particle():
    def __init__(self, gp_path):
        # self.gp = gp.SVGP(1000) # Does not work
        self.gp = gp.SVGP.load(1000, gp_path)

if __name__ == '__main__':
    Train_gps()