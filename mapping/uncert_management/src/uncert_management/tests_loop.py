#!/usr/bin/env python3
import rospy
import roslaunch
from std_msgs.msg import Bool


class ui_data_gen(object):

    def __init__(self):

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)

        self.synch_pub = rospy.Subscriber("/gt/ui_finished", Bool, self.synch_cb)
        self.finished = False

        for i in range(23, 28):

            cli_args = ['/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/mapping/uncert_management/launch/ui_test.launch', 'test:=' + str(i)]
            roslaunch_args = cli_args[1:]
            roslaunch_file = [
                (roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]

            parent = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
            print("Launching process ", i)
            parent.start()

            while not rospy.is_shutdown() and not self.finished:
                rospy.sleep(1)

            print("Shutting down process ", i)
            rospy.sleep(5)
            parent.shutdown()
            self.finished = False

    def synch_cb(self, finished_msg):
        self.finished = True


if __name__ == '__main__':

    rospy.init_node('ui_data_generation', disable_signals=False, anonymous=True)
    try:
        ui_data_gen()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not launch auv ui')
