#!/usr/bin/env python3
import rospy
from subprocess import call, Popen

class particles_launcher():
    def __init__(self):
        self.num_particles = rospy.get_param('~num_particles', 2) # Particle Count
        launch_file = rospy.get_param('~particle_launch_file', "particle.launch") # Particle Count
        
        print("Launching particles: ", self.num_particles)
        for i in range(0, self.num_particles):
            print("Launching particle: ", i)
            proc = Popen(["roslaunch", launch_file, "node_name:=particle_" + str(i)])

        rospy.spin()

if __name__ == '__main__':

    rospy.init_node('particle_launcher')

    try:
        launcher = particles_launcher()
        
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch rbpf_node")
