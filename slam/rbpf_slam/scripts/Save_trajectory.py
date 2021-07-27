#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool

class keyboard_trajectory():
    def __init__(self): 
        '''
        Control when to save trajectory and final map. 
        Open in a seperate terminal so one clearly can read 
        the output and control with the keyboard.
        '''

        pub_trajectory = rospy.Publisher('/keyboard_trajectory', Bool, queue_size=1)
        pub_map = rospy.Publisher('/gt/survey_finished', Bool, queue_size=1)
        msg = Bool()
        msg.data = True
        
        while True:
            val = input('Press  t  to save down the trajectory \nPress  m  to save down the final map \nPress  q  to quit\n...')
            if val == 'q':
                break
            elif val =='t' or val == 'T':
                pub_trajectory.publish(msg)
                print('Trajectory will be saved.')
            elif val =='m' or val == 'M':
                pub_map.publish(msg)
                print('Final map will be saved.\nTerminal shuts down.')
                break

            
        # rospy.spin()

if __name__ == '__main__':
    rospy.init_node('Save_trajectory', disable_signals=False)
    try:
        keyboard_trajectory()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch rbpf_node")
        pass