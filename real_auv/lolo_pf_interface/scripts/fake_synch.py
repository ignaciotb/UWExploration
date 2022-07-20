#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
import numpy as np
from std_msgs.msg import Bool

class FakeSynch:

    def __init__(self):

        odom_top_in = rospy.get_param("~odometry_topic", "/lolo/dr/odom")
        rospy.Subscriber(odom_top_in, Odometry, self.odom_cb, queue_size=100)

        finished_top = rospy.get_param("~survey_finished_top", "/finished")
        self.finish_pub = rospy.Publisher(finished_top, Bool, queue_size=0)

        self.time_odom = rospy.Time.now().to_sec()
        self.time = rospy.Time.now().to_sec()
        self.survey_finished = False
        self.survey_started = False
        
        while not rospy.is_shutdown():
            self.time = rospy.Time.now().to_sec()

            if self.time - self.time_odom > 6. and self.survey_started and not self.survey_finished:
                # Rosbag finished
                self.finish_pub.publish(True)
                self.survey_finished = True
                rospy.loginfo("-------------------ROSBAG FINISHED-------------------")

            rospy.sleep(0.1)

    def odom_cb(self, odom_t):
        if not self.survey_started:
            self.survey_started = True
        
        self.time_odom = rospy.Time.now().to_sec()


if __name__ == "__main__":
    rospy.init_node("fake_synch_node")

    fake_synch = FakeSynch()
