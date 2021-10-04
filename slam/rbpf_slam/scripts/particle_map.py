#!/usr/bin/env python3
import rospy
from bathy_gps.gp import SVGP # GP
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

import numpy as np


class particle_map(object):

    def __init__(self):
        self.gp = SVGP(50)

        self.storage_path = rospy.get_param("~results_path")
        self.count_training = 0

        self.node_name = rospy.get_name()

        # Subscription to training points 
        training_points = rospy.get_param("~training_points_top", 'mbes_pings')
        rospy.Subscriber(self.node_name + training_points, PointCloud2, self.train_map_cb, queue_size=2)

        rospy.spin()

    def train_map_cb(self, pings_msg):
        particle_number = self.node_name.split('_')[1]
        print("Training GP ", particle_number)
        beams = []
        for p in pc2.read_points(pings_msg, 
                                field_names = ("x", "y", "z"), skip_nans=True):
            beams.append(p)

        beams = np.asarray(beams)
        beams = np.reshape(beams, (-1,3))  

        self.gp.fit(beams[:,0:2], beams[:,2], n_samples= 500, 
                    max_iter=200, learning_rate=1e-1, rtol=1e-4, 
                    ntol=100, auto=False, verbose=False)
        # Plot posterior
        # self.gp.plot(beams[:,0:2], beams[:,2], 
        #              self.storage_path + 'gp_result/' + 'particle_' + str(particle_number) 
        #              + '_training_' + str(self.count_training) + '.png',
        #              n=100, n_contours=100 )

        print("GP trained ", particle_number)
        self.count_training += 1


if __name__ == '__main__':

    rospy.init_node('particle_map' , disable_signals=False)

    try:
        particle_map()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch particle_map")
