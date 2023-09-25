#!/usr/bin/env python3
import rospy
from subprocess import call, Popen
import numpy as np

class particles_launcher():
    def __init__(self):
        self.namespace = rospy.get_param('~namespace', "hugin")
        self.num_particle_hdl = rospy.get_param('~num_particle_handlers', 2)
        self.storage_path = rospy.get_param("~results_path", "./ros/")
        self.num_particles_per_hdl = rospy.get_param('~num_particles_per_handler', 2)
        launch_file = rospy.get_param('~particle_launch_file', "particle.launch")

        print("Launching particles: ", self.num_particle_hdl*self.num_particles_per_hdl)
        launchers_ids = np.linspace(0,self.num_particle_hdl*self.num_particles_per_hdl-self.num_particles_per_hdl,
                                    self.num_particle_hdl)

        for i in launchers_ids.astype(int):
            print("Launching particle handler: ", i)
            proc = Popen(["roslaunch", launch_file, "node_name:=particle_hdl_" + str(i),
                          "num_particles_per_handler:=" + str(self.num_particles_per_hdl),
                          "namespace:=" + str(self.namespace),
                          "storage_path:=" + str(self.storage_path)])
            # rospy.sleep(int(self.num_particles_per_hdl))
            rospy.sleep(3)

        rospy.spin()

if __name__ == '__main__':

    rospy.init_node('particle_launcher')

    try:
        launcher = particles_launcher()

    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch rbpf_node")
