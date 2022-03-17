#!/usr/bin/env python3

import rospy
import roslaunch
from std_msgs.msg import Bool
import os
import numpy as np
from pathlib import Path

class pf_data_gen(object):

    def __init__(self):

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)

        self.synch_pub = rospy.Subscriber("/gt/pf_finished", Bool, self.synch_cb)
        self.finished = False

        tests = [15, 16, 17, 14, 20] # UI
        path = "/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/utils/uw_tests/pf/overnight_2020/"
        # for std in np.linspace(4.,4.9,10):
        for std in [4.6]:
            Path(path + str(std)).mkdir(parents=True, exist_ok=True)
            for i in tests:
                cli_args = ['/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/localization/auv_particle_filter/launch/auv_pf.launch', 
                            'test:=' + str(i), 'measurement_std:=' + str(std), 'res_folder:=' + str(std)]
                roslaunch_args = cli_args[1:]
                roslaunch_file = [
                    (roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

                parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
                print("Launching test ", i, " with std ", std)
                parent.start()

                while not rospy.is_shutdown() and not self.finished:
                    rospy.sleep(1)

                print("Shutting down test ", i, " with std ", std)
                rospy.sleep(5)
                parent.shutdown()
                self.finished = False

        duration = 2  # seconds
        freq = 340  # Hz
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

    def synch_cb(self, finished_msg):
        self.finished = True


if __name__ == '__main__':

    rospy.init_node('pf_data_generation', disable_signals=False, anonymous=True)
    try:
        pf_data_gen()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not launch PF ')
