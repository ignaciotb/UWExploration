#!/usr/bin/env python

import rospy
import matplotlib.pyplot as plt
#  from cola2_msgs.msg import DVLBeam, DVL
import numpy as np
from std_msgs.msg import Float32

class DVLVisualization(object):
    
    def __init__(self):
        #  self.dvl_sub = rospy.Subscriber('/sam/core/dvl', DVL, self.dvl_cb)
        self.dvl_sub = rospy.Subscriber('/pf/n_eff', Float32, self.stat_cb)

        self.filter_cnt = 1
        self.filt_vec = np.zeros((1,1))
        rospy.spin()

    #  def dvl_cb(self, dvl_msg):
#
        #  vel_t = np.array([dvl_msg.velocity.x, dvl_msg.velocity.y, dvl_msg.velocity.z])
        #  self.filt_vec = np.hstack((self.filt_vec, vel_t.reshape(3,1)))
        #  self.filter_cnt += 1
        #  if self.filter_cnt > 0:
            #  # for stopping simulation with the esc key.
            #  plt.gcf().canvas.mpl_connect('key_release_event',
                    #  lambda event: [exit(0) if event.key == 'escape' else None])
#
            #  plt.subplot(3, 1, 1)
            #  plt.cla()
            #  plt.plot(np.linspace(0,self.filter_cnt, self.filter_cnt),
                     #  self.filt_vec[0,:], "-b")
            #
            #  plt.subplot(3, 1, 2)
            #  plt.cla()
            #  plt.plot(np.linspace(0,self.filter_cnt, self.filter_cnt),
                     #  self.filt_vec[1,:], "-b")
#
            #  plt.subplot(3, 1, 3)
            #  plt.cla()
            #  plt.plot(np.linspace(0,self.filter_cnt, self.filter_cnt),
                     #  self.filt_vec[2,:], "-b")
#
            #  plt.grid(True)
            #  plt.pause(0.001)
    
    def stat_cb(self, stat_msg):

        vel_t = np.array([stat_msg.data])
        self.filt_vec = np.hstack((self.filt_vec, vel_t.reshape(1,1)))
        self.filter_cnt += 1
        if self.filter_cnt > 0:
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])

            plt.subplot(1, 1, 1)
            plt.cla()
            plt.plot(np.linspace(0,self.filter_cnt, self.filter_cnt),
                     self.filt_vec[0,:], "-b")
            
            #  plt.subplot(3, 1, 2)
            #  plt.cla()
            #  plt.plot(np.linspace(0,self.filter_cnt, self.filter_cnt),
                     #  self.filt_vec[1,:], "-b")
#
            #  plt.subplot(3, 1, 3)
            #  plt.cla()
            #  plt.plot(np.linspace(0,self.filter_cnt, self.filter_cnt),
                     #  self.filt_vec[2,:], "-b")
#
            plt.grid(True)
            plt.pause(0.001)



if __name__ == "__main__":
    rospy.init_node("dvl_visual")
    try:
        DVLVisualization()
    except rospy.ROSInterruptException:
        pass
