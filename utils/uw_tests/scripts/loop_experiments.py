#!/usr/bin/env python3

import rospy
import roslaunch
from std_msgs.msg import Bool
# import os
import numpy as np
from pathlib import Path

class experiments_loop(object):

    def __init__(self):

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)

        finished_top = rospy.get_param("~rbpf_saved_top", "/gt/rbpf_saved")
        self.synch_pub = rospy.Subscriber(finished_top, Bool, self.synch_cb)
        self.finished = False
        dataset = "lolo_0"
        particle_count = 100
        num_particle_handlers = 10
        path = "/media/orin/Seagate Expansion Drive/rbpf_results/lolo_0/"

        tests = [4] # UI
        # for std in np.linspace(4.,4.9,10):
        # for std in [0]:
        for i in tests:
            Path(path + str(i)).mkdir(parents=True, exist_ok=True)
            cli_args = ['/home/orin/catkin_ws/src/UWExploration/slam/rbpf_slam/launch/rbpf_slam.launch', 
                        'num_particle_handlers:=' + str(num_particle_handlers), 'particle_count:=' + str(particle_count),
                        'results_path:=' + path + str(i), "dataset:=" + str(dataset)]
            roslaunch_args = cli_args[1:]
            roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], 
                                roslaunch_args)]

            parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
            print("Launching test ", i)
            parent.start()

            while not rospy.is_shutdown() and not self.finished:
                rospy.sleep(1)

            print("Shutting down test ", i)
            # rospy.sleep(particle_count*10)
            parent.shutdown()
            self.finished = False

        # duration = 2  # seconds
        # freq = 340  # Hz
        # os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

    def synch_cb(self, finished_msg):
        self.finished = True


if __name__ == '__main__':

    rospy.init_node('experiments_loop_node', disable_signals=False, anonymous=True)
    try:
        experiments_loop()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not launch experiments')
