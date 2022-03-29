#!/usr/bin/env python3
import rospy
# from bathy_gps.gp import SVGP # GP
from gp_mapping import gp
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from slam_msgs.msg import PlotPosteriorResult, PlotPosteriorAction
from slam_msgs.msg import SamplePosteriorResult, SamplePosteriorAction

import actionlib

import numpy as np


class particle_map(object):

    def __init__(self):
        self.gp = gp.SVGP(50) # num of inducing points

        self.storage_path = rospy.get_param("~results_path")
        self.count_training = 0
        
        self.node_name = rospy.get_name()
        self.particle_number = self.node_name.split('_')[1]

        # Subscription to GP training points 
        training_points = rospy.get_param("~training_points_top")
        rospy.Subscriber(self.node_name + training_points, PointCloud2, self.train_map_cb, queue_size=2)

        # Service for plotting results
        plot_gp_name = rospy.get_param("~plot_gp_server")
        self._as_plot = actionlib.SimpleActionServer(self.node_name + plot_gp_name, PlotPosteriorAction, 
                                                execute_cb=self.plot_posterior, auto_start = False)
        self._as_plot.start()

        # Service for expected meas
        sample_gp_name = rospy.get_param("~sample_gp_server")
        self._as_sample = actionlib.SimpleActionServer(self.node_name + sample_gp_name, SamplePosteriorAction, 
                                                execute_cb=self.sample_posterior, auto_start = False)
        self._as_sample.start()

        self.training = False
        self.plotting = False

        rospy.spin()

    def sample_posterior(self, goal):

        beams = []
        for p in pc2.read_points(goal.ping, 
                                field_names = ("x", "y", "z"), skip_nans=True):
            beams.append(p)

        beams = np.asarray(beams)
        beams = np.reshape(beams, (-1,3)) 

        while self.training:
            rospy.Rate(1).sleep()
            print("GP ", self.particle_number, " waiting for training before sampling")

        mu, sigma = self.gp.sample(np.asarray(beams)[:, 0:2])

        # Set action as success
        result = SamplePosteriorResult()
        result.mu = mu
        result.sigma = sigma
        self._as_sample.set_succeeded(result)
        # self.plotting = False
        print("GP ", self.particle_number, " sampled")


    def plot_posterior(self, goal):

        while self.training:
            rospy.Rate(1).sleep()
            print("GP ", self.particle_number, " waiting for training before plotting")

        beams = []
        for p in pc2.read_points(goal.pings, 
                                field_names = ("x", "y", "z"), skip_nans=True):
            beams.append(p)

        beams = np.asarray(beams)
        beams = np.reshape(beams, (-1,3)) 

        # Plot posterior and save it to image
        print("Plotting GP ", self.particle_number)
        self.plotting = True
        self.gp.plot(beams[:,0:2], beams[:,2], 
                     self.storage_path + 'particle_' + str(self.particle_number) 
                     + '_training_' + str(self.count_training-1) + '.png',
                     n=50, n_contours=100 )

        # Set action as success
        result = PlotPosteriorResult()
        result.success = True
        self._as_plot.set_succeeded(result)
        # self.plotting = False
        print("GP plotted ", self.particle_number)


    def train_map_cb(self, pings_msg):

        # If plotting, the mission has ended
        if not self.plotting:
            
            beams = []
            for p in pc2.read_points(pings_msg, 
                                    field_names = ("x", "y", "z"), skip_nans=True):
                beams.append(p)

            beams = np.asarray(beams)
            beams = np.reshape(beams, (-1,3))  

            print("Training GP ", self.particle_number)
            self.training = True
            self.gp.fit(beams[:,0:2], beams[:,2], n_samples= 200, 
                        max_iter=200, learning_rate=1e-1, rtol=1e-4, 
                        n_window=100, auto=False, verbose=False)

            # # Plot posterior and save it to image
            # Uncomment this when running with one particle to plot maps after training
            # self.gp.plot(beams[:,0:2], beams[:,2], 
            #              self.storage_path + 'gp_result/' + 'particle_' + str(self.particle_number) 
            #              + '_training_' + str(self.count_training) + '.png',
            #              n=100, n_contours=100 )

            print("GP trained ", self.particle_number)
            self.count_training += 1
            self.training = False


if __name__ == '__main__':

    rospy.init_node('particle_map' , disable_signals=False)

    try:
        particle_map()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch particle_map")
