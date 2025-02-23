#!/usr/bin/env python3

import rospy
import numpy as np
from auv_model.msg import Sidescan
import cv2

class SSSVisualizer:

    def __init__(self):
        rospy.init_node('view_sidescan_particles', anonymous=True)
        
        self.num_subscribers = rospy.get_param('~num_subscribers', 1)
        self.topic_name = rospy.get_param('~topic_name', 'chatter')
        
        self.subscribers = []
        self.data = [np.zeros((1000, 1000), dtype=np.ubyte) for _ in range(self.num_subscribers)]
        
        for i in range(self.num_subscribers):
            topic = f"{self.topic_name}" + "/particle_" + str(i)
            sub = rospy.Subscriber(topic, Sidescan, self.sss_callback, callback_args=i, queue_size=100)
            self.subscribers.append(sub)
            rospy.loginfo(f"Subscribed to {topic}")
        
        while not rospy.is_shutdown():
            self.update_display()
            rospy.Rate(0.1).sleep()

    def sss_callback(self, msg, index):
        
        port = np.array(bytearray(msg.port_channel), dtype=np.ubyte)
        stbd = np.array(bytearray(msg.starboard_channel), dtype=np.ubyte)
        meas = np.concatenate([np.flip(port), stbd])
        self.data[index][1:, :] = self.data[index][:-1, :]
        self.data[index][0, :] = meas

        self.update_display()
    
    def update_display(self):
        
        # Determine grid layout based on number of subscribers
        rows = int(np.ceil(np.sqrt(self.num_subscribers)))
        cols = rows + self.num_subscribers % rows
        
        white_margin = np.ones((10, 1000 * cols), dtype=np.uint8) * 255  # White margin between rows
        display_img_rows = []
        for row in range(0, int(len(self.data)/cols)):
            display_img_rows.append(cv2.hconcat(self.data[row*(cols): row*(cols)+cols]))
            if row < rows - 2:
                display_img_rows.append(white_margin)

        display_img = display_img_rows[0]
        for row in range(1, len(display_img_rows)):
            display_img = cv2.vconcat([display_img, display_img_rows[row]])

        display_img_resized = cv2.resize(display_img, (1920, 1080))
        cv2.imshow("Sidescan images", display_img_resized)
        cv2.waitKey(1)


if __name__ == '__main__':
    
    try:
        node = SSSVisualizer()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
