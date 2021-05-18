#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool

class keyboard_trajectory():
    def __init__(self): 
        # Control when to save trajectory
        pub_ = rospy.Publisher('/keyboard_trajectory', Bool, queue_size=1)
        print('\n(Press q to quit)\n')
        msg = Bool()
        msg.data = True
        
        while True:
            v = input('Press any key to save down trajectory ...')
            if v == 'q':
                break
            pub_.publish(msg)
            print('Trajectory will be saved.')
            
        # rospy.spin()

if __name__ == '__main__':
    rospy.init_node('Save_trajectory', disable_signals=False)
    try:
        keyboard_trajectory()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch rbpf_node")
        pass