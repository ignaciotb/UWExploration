#!/usr/bin/python

# Copyright 2019 Ignacio Torroba (torroba@kth.se)
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import pygame
from pygame.constants import K_LEFT, K_RIGHT, K_UP, K_DOWN, K_w, K_s, K_z, K_a, K_d, K_m, K_n
import rospy
import numpy as np
from std_msgs.msg import Header, Float64, Bool

class AUVTeleopServer(object):

    def __init__(self):

        pygame.init()

        self.throttle_top = rospy.get_param('~throttle_cmd', '/throttle')
        self.thruster_top = rospy.get_param('~thruster_cmd', '/thruster')
        self.inclination_top = rospy.get_param('~inclination_cmd', '/inclination')

        throttle_pub = rospy.Publisher(self.throttle_top, Float64, queue_size=1)
        thruster_pub = rospy.Publisher(self.thruster_top, Float64, queue_size=1)
        inclination_pub = rospy.Publisher(self.inclination_top, Float64, queue_size=1)

        screen = pygame.display.set_mode((199, 200))
        pygame.display.flip()
        header = Header()

        thruster_angle = 0.2
        inclination_angle = 0.2
        throttle_level = 5.

        clock = pygame.time.Clock()
        while not rospy.is_shutdown():

            keys = pygame.key.get_pressed()
            incl = Float64()
            throttle = Float64()
            thrust = Float64()

            thrust.data = 0.0
            incl.data = 0.0
            throttle.data = 0.0

            # Steering
            if keys[K_LEFT]:
                thrust.data = thruster_angle
            if keys[K_RIGHT]:
                thrust.data = -thruster_angle
            if keys[K_UP]:
                incl.data = -inclination_angle
            if keys[K_DOWN]:
                incl.data = inclination_angle
            thruster_pub.publish(thrust)
            inclination_pub.publish(incl)

            # Thrusting
            if keys[K_w]:
                throttle.data = throttle_level
                throttle_pub.publish(throttle)
            if keys[K_s]:
                throttle.data = -throttle_level
                throttle_pub.publish(throttle)

            pygame.event.pump()
            #  rospy.sleep(0.1)
            clock.tick(10)

if __name__ == "__main__":

    rospy.init_node('auv_keyboard_teleop')

    try:
        AUVTeleopServer()
    except rospy.ROSInterruptException:
        pass
