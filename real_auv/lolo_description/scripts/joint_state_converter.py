#!/usr/bin/env python

import rospy
import math
from smarc_msgs.msg import FloatStamped, ThrusterFeedback
from sensor_msgs.msg import JointState

class LoloJointStateConverter(object):

    def elevon_port_callback(self, msg):
        state = JointState()
        state.name = ["lolo/elevon_port_joint"]
        state.position = [-msg.data]
        self.joint_state_pub.publish(state)

    def elevon_stbd_callback(self, msg):
        state = JointState()
        state.name = ["lolo/elevon_stbd_joint"]
        state.position = [-msg.data]
        self.joint_state_pub.publish(state)

    def rudder_callback(self, msg):
        state = JointState()
        state.name = ["lolo/rudder_port_joint", "lolo/rudder_stbd_joint"]
        state.position = [msg.data, msg.data]
        self.joint_state_pub.publish(state)

    def elevator_callback(self, msg):
        state = JointState()
        state.name = ["lolo/elevator_joint"]
        state.position = [-msg.data]
        self.joint_state_pub.publish(state)

    def thruster_port_callback(self, msg):
        state = JointState()
        state.name = ["lolo/thruster_joint_port"]
        self.velocities[0] = -0.1 * 2.*math.pi/60.*float(msg.rpm.rpm)
        state.velocity = [self.velocities[0]]
        self.joint_state_pub.publish(state)

    def thruster_stbd_callback(self, msg):
        state = JointState()
        state.name = ["lolo/thruster_joint_stbd"]
        self.velocities[1] = 0.1 * 2.*math.pi/60.*float(msg.rpm.rpm)
        state.velocity = [self.velocities[1]]
        self.joint_state_pub.publish(state)

    def timer_callback(self, event):
        state = JointState()
        state.name = ["lolo/thruster_port_joint", "lolo/thruster_stbd_joint"]
        duration = (rospy.Time.now() - self.start_time).to_sec()
        state.position = [duration*vel for vel in self.velocities]
        self.joint_state_pub.publish(state)

    def __init__(self):
        self.joint_state_pub = rospy.Publisher("command_states", JointState, queue_size=10)

        self.elevon_stbd_sub = rospy.Subscriber("core/elevon_strb_fb", FloatStamped, self.elevon_stbd_callback)
        self.elevon_port_sub = rospy.Subscriber("core/elevon_port_fb", FloatStamped, self.elevon_port_callback)
        self.elevator_sub = rospy.Subscriber("core/elevator_fb", FloatStamped, self.elevator_callback)
        self.rudder_sub = rospy.Subscriber("core/rudder_fb", FloatStamped, self.rudder_callback)

        #self.thruster_stbd_sub = rospy.Subscriber("core/thruster_strb_fb/rpm", ThrusterFeedback, self.thruster_stbd_callback)
        #self.thruster_port_sub = rospy.Subscriber("core/thruster_port_fb/rpm", ThrusterFeedback, self.thruster_port_callback)
        self.thruster_port_sub = rospy.Subscriber("core/thruster1_fb", ThrusterFeedback, self.thruster_port_callback)
        self.thruster_stbd_sub = rospy.Subscriber("core/thruster2_fb", ThrusterFeedback, self.thruster_stbd_callback)

        self.start_time = rospy.Time.now()
        self.velocities = [0., 0.]
        self.timer = rospy.Timer(rospy.Duration(0.1), self.timer_callback)

if __name__ == '__main__':
    rospy.init_node('joint_state_converter')
    converter = LoloJointStateConverter()
    rospy.spin()
