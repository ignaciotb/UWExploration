#!/usr/bin/env python3

import rospy
import torch
import time
from sensor_msgs.msg import PointCloud2
import matplotlib.pyplot as plt
from bathy_gps.gp import SVGP
import numpy as np
import os
from auvlib.data_tools import benchmark
from auvlib.data_tools import std_data, xyz_data
from scipy.spatial.transform import Rotation as rot
from std_msgs.msg import Float32, Header, Bool, Float32MultiArray, ByteMultiArray
from rospy_tutorials.msg import Floats
from rospy.numpy_msg import numpy_msg

# train in a new node to save time 
class Train_gps():

    def __init__(self):        
        # ---------------- publish / subscribe ------------ 
        self.pc = rospy.get_param('~particle_count', 10) # Particle Count
        self.n_inducing = rospy.get_param("~n_inducing", 300) # Number of inducing points and optimisation samples 
        self.storage_path = rospy.get_param("~data_path") #'/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/data/results/'
        self.firstFit = [True] * self.pc # len of particles
        self.count_training = [0] * self.pc # len of particles
        self.numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        self.gp_obj = np.empty(self.pc, dtype=object)
        for i in range(0,self.pc):
            self.gp_obj[i] = create_particle(self.n_inducing)
        # Subscribe to particles
        rospy.Subscriber('/training_gps', numpy_msg(Floats), self.cb, queue_size=10)
        # Publish variance and mean
        self.meanvar_pub = rospy.Publisher('/gp_meanvar', numpy_msg(Floats),queue_size=10)
        time.sleep(5)
        rospy.loginfo('Patiently waiting for data...')
        rospy.spin()


        # --------------- working on its own -------------
        # self.data_path = '/home/stine/catkin_ws/src/UWExploration/slam/rbpf_slam/data/results/'
        # # _, _, self.names = next(os.walk(self.data_path + 'particles/'))
        # self.pc = 1 #len(self.names)
        # self.particle = np.empty(self.pc, dtype=object)
        # self.n = 300
        # self.trainNplot()
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

            # n_samples need to be smaller than self.n, if max_iter is too large it will lead to overfitting
            self.particle[i].gp.fit(inputs, targets, n_samples= int(self.n/2), max_iter=int(self.n/2), learning_rate=1e-1, rtol=1e-4, ntol=100, auto=False, verbose=True)
            print('\nParticle {} done ({} in total) \n'.format(i, self.pc))
            self.particle[i].gp.plot(inputs, targets, self.data_path + 'training.png', n=100, n_contours=100 )
            gp_path = self.data_path + 'svgp.pth'
            # self.particle[i].gp.save(gp_path)
            # save the posterior point cloud
            # self.plot_and_save_posterior(inputs, targets, gp_path)

    def plot_and_save_posterior(self, inputs, targets, path):
        x = inputs[:,0]
        y = inputs[:,1]
        gp = SVGP.load(self.n, path )
        gp.plot(inputs, targets, self.data_path + 'posterior.png')
        gp.save_posterior(self.n, min(x), max(x), min(y), max(y), self.data_path + 'posterior.npy', verbose=False)


    def rms(self, ping_cloud):
        gp_cloud = np.load(self.storage_path + 'posterior.npy')
        # ping_cloud =  np.load(self.data_path + 'ping_cloud.npy')

        cloud = [ping_cloud, gp_cloud[:,0:3]]

        # Ground truth benchmark
        bm = benchmark.track_error_benchmark()
        bm.add_ground_truth(cloud, cloud)

        ping_list = [ping_cloud]
        bm.add_benchmark(ping_list, ping_list, "real")

        gp_list = [gp_cloud[:,0:3]]
        bm.add_benchmark(gp_list, gp_list, "gp")


        bm.print_summary()

    def cb(self, msg):
        arr = msg.data
        p_id = int(arr[-1]) # Particle ID
        # print('training particle ', p_id)
        arr = np.delete(arr, -1)
        n = int(arr.shape[0] / 3)
        cloud = arr.reshape(n,3)
        inputs = cloud[:,[0,1]]
        targets = cloud[:,2]

        # self.train2(inputs, targets, p_id)
        self.trainGP(inputs, targets, p_id)

    def train2(self, inputs, targets, p_id):
        if self.firstFit[p_id]: # Only enter ones
            self.firstFit[p_id] = False 
            gp = SVGP(self.n_inducing)
            # train each particles gp
            gp.fit(inputs, targets, n_samples= int(self.n_inducing/2), max_iter=int(self.n_inducing/2), learning_rate=1e-1, rtol=1e-4, ntol=100, auto=False, verbose=True)
            # save a plot of the gps
            gp.plot(inputs, targets, self.storage_path + 'particle' + str(p_id) + 'training' + str(self.count_training[p_id]) + '.png', n=100, n_contours=100 )
            # save the path to train again
            gp_path = self.storage_path + 'svgp_particle' + str(p_id) + '.pth'
            gp.save(gp_path)
        
        else: # second or more time to retrain gp
            gp_path = self.storage_path + 'svgp_particle' + str(p_id) + '.pth'
            gp = SVGP.load(self.n_inducing, gp_path)
            # train each particles gp
            gp.fit(inputs, targets, n_samples= int(self.n_inducing/2), max_iter=int(self.n_inducing/2), learning_rate=1e-1, rtol=1e-4, ntol=100, auto=False, verbose=True)
            # save a plot of the gps
            gp.plot(inputs, targets, self.storage_path + 'particle' + str(p_id) + 'training' + str(self.count_training[p_id]) + '.png', n=100, n_contours=100 )
            # save the path to train again
            gp_path = self.storage_path + 'svgp_particle' + str(p_id) + '.pth'
            gp.save(gp_path)
        
        print('\n... done with particle {} training {} '.format(p_id , self.count_training[p_id]))
        self.count_training[p_id] +=1 # to save plots

    def trainGP(self, inputs, targets, p_id):
        # train each particles gp
        try:
            self.gp_obj[p_id].gp.fit(inputs, targets, n_samples= int(self.n_inducing/2), max_iter=int(self.n_inducing/2), learning_rate=1e-1, rtol=1e-4, ntol=100, auto=False, verbose=False)
            # if self.count_training[p_id] > 0 and self.count_training[p_id] % 10 == 0:
            #     # save a plot of the gps
            #     # rospy.loginfo('Saving a plot of the gps')
            #     self.gp_obj[p_id].gp.plot(inputs, targets, self.storage_path + 'particle' + str(p_id) + 'training' + str(self.count_training[p_id]) + '.png', n=100, n_contours=100 )
            # print('\n ... saving the posterior...')
            # x = inputs[:,0]
            # y = inputs[:,1]
            # self.gp_obj[p_id].gp.save_posterior(self.n_inducing, min(x), max(x), min(y), max(y), self.storage_path + 'particle' + str(p_id) + 'posterior.npy', verbose=False)
            # if self.count_training[p_id] < len(self.numbers):
            #     print('... done with particle {} training {} '.format(p_id , self.numbers[self.count_training[p_id]]))
            # else:
            #     print('... done with particle {} training {} '.format(p_id , self.count_training[p_id]))

            self.count_training[p_id] +=1 # to save plots

            #  publish results ---------
            mean, variance = self.gp_obj[p_id].gp.sample(inputs)
            arr = np.zeros((len(mean),2))
            arr[:,0] = mean
            arr[:,1] = variance
            arr = arr.reshape(len(mean)*2, 1)    
        except:
            print('gitter..')
            arr = np.array([0, 0])
            
        arr = np.append(arr, p_id) # insert particle index
        msg = Floats()
        msg.data = arr
        self.meanvar_pub.publish(msg)
        

class create_particle():
    def __init__(self, n):
        self.gp = SVGP(n) 
        # self.gp = SVGP.load(1000, gp_path)




if __name__ == '__main__':
    rospy.init_node('Train_gps_node', disable_signals=False)
    try:
        Train_gps()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch rbpf_node")
        pass
    # Train_gps()
    # print('\ntraining done')
    # print('\nnow time for rms\n')
    # working.rms()