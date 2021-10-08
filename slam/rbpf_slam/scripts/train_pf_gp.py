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
        qz = rospy.get_param("~queue_size", 10) # to pub/sub to gp training
        self.l_max = rospy.get_param("~l_max", 20.)

        self.firstFit = [True] * self.pc # len of particles
        self.count_training = [0] * self.pc # len of particles
        self.numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
        self.gp_obj = np.empty(self.pc, dtype=object)

        if not os.path.exists(self.storage_path + 'gp_result/'):
            os.makedirs(self.storage_path + 'gp_result/')
        self.f = open(self.storage_path + 'gp_result/gp_training.txt', 'w+')
        # Subscribe to particles
        rospy.Subscriber('/training_gps', numpy_msg(Floats), self.cb, queue_size=qz)
        # Publish variance and mean
        self.meanvar_pub = rospy.Publisher('/gp_meanvar', numpy_msg(Floats),queue_size=qz)
        # check length scale
        # rospy.Subscriber('/length_scale', Bool, self.cb_lengthscale, queue_size=1)
        self.length_pub = rospy.Publisher('/length_scale', numpy_msg(Floats), queue_size=qz)
        time.sleep(10)
        # initialize gps for each particle
        for i in range(0,self.pc):
            self.gp_obj[i] = create_particle(self.n_inducing)
            self.check_lengthscale(i)
        
        rospy.loginfo('Patiently waiting for data...')
        rospy.spin()

    def cb(self, msg):
            arr = msg.data
            final = int(arr[-1]) # if final or not
            arr = np.delete(arr, -1)
            idx = int(arr[-1]) # Particle index
            # print('training particle ', idx)
            arr = np.delete(arr, -1)
            n = int(arr.shape[0] / 3)
            cloud = arr.reshape(n,3)
            inputs = cloud[:,[0,1]]
            targets = cloud[:,2]
            self.trainGP(inputs, targets, idx, final)

    def trainGP(self, inputs, targets, idx, final):
        t0 = time.time()
        # train each particles gp
        try:
            self.gp_obj[idx].gp.fit(inputs, targets, n_samples= int(self.n_inducing/2), max_iter=int(self.n_inducing/4), learning_rate=1e-1, rtol=1e-4, ntol=100, auto=False, verbose=False)
            # check length scale
            self.gp_obj[idx].update_lengthscale()
            self.check_lengthscale(idx)

            # print('for particle ', idx , ' this is the lengthclale (training) \n', self.gp_obj[idx].gp.cov.lengthscale )
            # if (idx < 1) and len(targets) > 5: # and self.count_training[idx] % 10 == 0:
            #     # save a plot of the gps every 10th training of the first and second particle
            #     rospy.loginfo('Saving a plot of the gps')
            #     self.gp_obj[idx].gp.plot(inputs, targets, self.storage_path + 'gp_plot/' + 'particle' + str(idx) + 'training' + str(self.count_training[idx]) + '.png', n=100, n_contours=100 )
            # print('\n ... saving the posterior...')
            
            if self.count_training[idx] < len(self.numbers):
                print('... done with particle {} training {} '.format(idx , self.numbers[self.count_training[idx]]))
                self.f.write('Particle {}   Training {}    Time {:.1f} seconds.\n'.format(idx , self.numbers[self.count_training[idx]], time.time() - t0))
            else:
                # print('... done with particle {} training {} '.format(idx , self.count_training[idx]))
                self.f.write('Particle {}   Training {}    Time {:.1f} seconds.\n'.format(idx , self.count_training[idx], time.time() - t0))


            # print('Training took {:.1f} seconds'. format(time.time() - t0))
            self.count_training[idx] +=1 # to save plots

            #  publish results ---------
            mean, variance = self.gp_obj[idx].gp.sample(inputs)
            arr = np.zeros((len(mean),2))
            arr[:,0] = mean
            arr[:,1] = variance
            arr = arr.reshape(len(mean)*2, 1)    
        except:
            rospy.loginfo('gitter..')
            arr = np.array([0, 0])
            
        if final == 99:
            rospy.loginfo('final gp running ...')
            x = inputs[:,0]
            y = inputs[:,1]
            self.gp_obj[idx].gp.save_posterior(self.n_inducing, min(x), max(x), min(y), max(y), self.storage_path + 'gp_result/' + 'particle' + str(idx) + 'posterior.npy', verbose=False)
            self.gp_obj[idx].gp.plot(inputs, targets, self.storage_path + 'gp_result/' + 'particle' + str(idx) + 'training' + str(self.count_training[idx]) + '.png', n=100, n_contours=100 )
            rospy.loginfo('final gp saved.')
            self.f.close()

        arr = np.append(arr, idx) # insert particle index
        msg = Floats()
        msg.data = arr
        self.meanvar_pub.publish(msg)

    def check_lengthscale(self, idx):
        msg = Floats()

        if self.gp_obj[idx].lengthscale < self.l_max:
            arr = np.array([0, idx]) # True = 0
        else: 
            print('lengthscale is above maximum value at:', self.gp_obj[idx].lengthscale)
            print('... for particle {} training {} '.format(idx , self.count_training[idx]))
            arr = np.array([1, idx]) # False = 1

        msg.data = arr
        # print('publish lengthscale ')
        self.length_pub.publish(msg)            


class create_particle():
    def __init__(self, n):
        self.gp = SVGP(n) 
        self.lengthscale = self.gp.cpu().cov.base_kernel.lengthscale.detach().numpy()[0,0]
    
    def update_lengthscale(self):
        mini = self.gp.cov.base_kernel.lengthscale.detach().numpy()[0,0]
        maxi = self.gp.cov.base_kernel.lengthscale.detach().numpy()[0,1]
        self.lengthscale = (mini + maxi)/2


if __name__ == '__main__':
    rospy.init_node('Train_gps_node', disable_signals=False)
    try:
        Train_gps()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch rbpf_node")
        pass