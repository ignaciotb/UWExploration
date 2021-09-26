#!/usr/bin/env python3

import rospy
from std_msgs.msg import Bool, Float32

class keyboard_trajectory():
    def __init__(self): 
        '''
        Control when to save trajectory and final map. 
        Open in a seperate terminal so one clearly can read 
        the output and control with the keyboard.
        '''

        pub_trajectory = rospy.Publisher('/keyboard_trajectory', Bool, queue_size=1)
        pub_map = rospy.Publisher('/gt/survey_finished', Bool, queue_size=1)
        pub_gp = rospy.Publisher('/final_gp_topic', Float32, queue_size=1)

        msg = Bool()
        msg.data = True

        while True:
            val = input('Press  m  to save down the xy-path \nPress  t  to save down the trajectory \nPress  q  to quit \nWrite the particle number which final map should be saved:\n...')
            if val == 'q':
                break
            elif val =='t' or val == 'T':
                pub_trajectory.publish(msg)
                print('Trajectory will be saved.\n')
            elif val =='m' or val == 'M':
                pub_map.publish(msg)
                print('Final xy-path will be saved.\n')
            elif int(val) > 0 and int(val) < 50: # Particle count
                print('choosing particle ', val)
                msgInt = Float32()
                msgInt.data = float(val)
                pub_gp.publish(msgInt)
                break


if __name__ == '__main__':
    rospy.init_node('Save_trajectory', disable_signals=False)
    try:
        keyboard_trajectory()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch rbpf_node")
        pass