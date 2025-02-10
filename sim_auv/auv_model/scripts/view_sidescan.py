#!/usr/bin/env python3
import rospy
import cv2
from smarc_msgs.msg import Sidescan
from functools import partial
import numpy as np
from ctypes import cast, pointer, POINTER, c_char, c_int


def convert(c):
    return cast(pointer(c_char(c)), POINTER(c_int)).contents.value


def callback(img, msg):

    #print msg

    #for p in msg.sidescan.sidescan.port_channel:
    #    print convert(p)

    port = np.array(bytearray(msg.port_channel), dtype=np.ubyte)
    stbd = np.array(bytearray(msg.starboard_channel), dtype=np.ubyte)
    #port = np.array([int(p) for p in msg.sidescan.sidescan.port_channel]) # dtype=np.ubyte)
    #stbd = np.array([int(p) for p in msg.sidescan.sidescan.starboard_channel])
    #stbd = np.array(msg.sidescan.sidescan.starboard_channel, dtype=float) #dtype=np.ubyte)
    #print port.shape, stbd.shape
    #print port, stbd
    meas = np.concatenate([np.flip(port), stbd])
    #print(meas)
    img[1:, :] = img[:-1, :]
    img[0, :] = meas


rospy.init_node('sidescan_viewer', anonymous=True)

img = np.zeros((1000, 1000), dtype=np.ubyte)  # dtype=float) #
cv2.namedWindow('Sidescan image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Sidescan image', 2 * 256, 1000)

rospy.Subscriber("/sam/sim/sss_pings", Sidescan, partial(callback, img))

# spin() simply keeps python from exiting until this node is stopped
r = rospy.Rate(5)  # 10hz
while not rospy.is_shutdown():
    resized = cv2.resize(img, (2 * 256, 1000), interpolation=cv2.INTER_AREA)
    cv2.imshow("Sidescan image", resized)
    cv2.waitKey(1)
    r.sleep()
