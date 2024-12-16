#!/usr/bin/env python3

# Standard dependencies
import math
import rospy
import sys
import numpy as np
# from scipy.spatial.transform import Rotation as rot

from tf.transformations import quaternion_matrix, euler_from_quaternion
from tf.transformations import rotation_matrix

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from scipy.ndimage import gaussian_filter1d

# For sim mbes action client
import actionlib
from auv_model.msg import MbesSimAction, MbesSimResult, SssSimAction, SssSimResult, Sidescan

# Auvlib
from auvlib.bathy_maps import base_draper
from auvlib.data_tools import csv_data, xtf_data

class sss_model(object):

    def __init__(self):

        # - For Dec 2024, Asko 
        # [estimated using logged temperature from DVL from Dec and measured SVPs from Oct 2024 (assuming salinity is the same)]
        # 1431.1 m/s is assuming salinity doesn't change 
        # 1430.1 m/s is eyeballing from the altitude (using mesh) and nadir line in the sidescan waterfall image

        self.svp = 1431.1 - 1 # mean sound speed (m/s) 
        isovelocity_sound_speeds = [csv_data.csv_asvp_sound_speed()]
        isovelocity_sound_speeds[0].dbars = np.arange(50)
        isovelocity_sound_speeds[0].vels = np.ones((50))*self.svp # m/s
        self.max_r = 40 # this is the number we've put in the sss driver when running the mission
        self.max_r = self.max_r/1500.0*self.svp # this is the actual max range accounting for the sound speed
        self.nbr_bins = 1000
        slant_range = np.arange(self.nbr_bins)*self.max_r/self.nbr_bins
        theta_port = np.ones_like(slant_range)*np.pi/2
        theta_stbd = -np.ones_like(slant_range)*np.pi/2

        # for TVG residual (iSSS)
        ab_poly =  [ 2.24377726e-04, -5.62777122e-03,  5.51645115e-02,  4.02042166e-01]
        self.TVG_residual_poly = np.polyval(ab_poly, slant_range)
        self.TVG_residual_poly[self.TVG_residual_poly>4]=4

        mesh_path = rospy.get_param('~mesh_path')
        data = np.load(mesh_path)
        V, F, bounds = data['V'], data['F'], data['bounds']
        print("Mesh loaded")

        # Create draper
        self.draper = base_draper.BaseDraper(V, F, bounds, isovelocity_sound_speeds)
        self.draper.set_sidescan_yaw(0)

        # Set sensor offset for sam here
        # # this is for iSSS stick
        # lever_arm = np.array([0.0,0.0,-1.6])

        # # this is because the orthometric height from RTK GPS uses a different geoid model than the one used in MBES
        # undulation_offset = np.array([0.0,0.0,1.4151990000000012])
        # sensor_offset =  lever_arm + undulation_offset
        # # print(f"appliying sensor offset: {sensor_offset}")
        # draper.set_sidescan_port_stbd_offsets(sensor_offset, sensor_offset)

        V = None
        F = None
        bounds = None
        print("draper created")
        print("Size of draper: ", sys.getsizeof(self.draper))

        # Action server for MBES pings sim (necessary to be able to use UFO maps as well)
        sim_sss_as = rospy.get_param('~sss_sim_as', '/sss_sim_server')
        server_mode = rospy.get_param("~server_mode", False)
        self.as_ping = actionlib.SimpleActionServer(sim_sss_as, SssSimAction,
                                                    execute_cb=self.sss_as_cb, auto_start=server_mode)

        rospy.spin()


    # Action server to simulate SSS for the sim AUV
    def sss_as_cb(self, goal):

        # Unpack goal
        p_sss = [goal.mbes_pose.transform.translation.x,
                  goal.mbes_pose.transform.translation.y,
                  goal.mbes_pose.transform.translation.z]
        euler_sss = euler_from_quaternion([goal.mbes_pose.transform.rotation.x,
                                            goal.mbes_pose.transform.rotation.y,
                                            goal.mbes_pose.transform.rotation.z,
                                            goal.mbes_pose.transform.rotation.w])
        
        # Create ping to render SSS
        xtf_ping = xtf_data.xtf_sss_ping()
        xtf_ping.port.time_duration = self.max_r*2/self.svp
        xtf_ping.stbd.time_duration = self.max_r*2/self.svp
        xtf_ping.pos_ = np.array(p_sss)
        xtf_ping.roll_, xtf_ping.pitch_, xtf_ping.yaw_ = euler_sss

        # Rendering
        left, right = self.draper.project_ping(xtf_ping, self.nbr_bins) # project
        # mbes = mbes[::-1]  # Reverse beams for same order as real pings

        sss_msg = Sidescan()

        sss_msg.port_channel = np.copy(np.array(left.time_bin_model_intensities)*255.)
        sss_msg.starboard_channel = np.copy(np.array(right.time_bin_model_intensities)*255.)
        sss_msg.port_channel[np.isnan(sss_msg.port_channel)] = 0
        sss_msg.starboard_channel[np.isnan(sss_msg.starboard_channel)] = 0

        ## if we want to apply TVG residual
        # sss_msg.port_channel = sss_msg.port_channel * self.TVG_residual_poly
        # sss_msg.starboard_channel = sss_msg.starboard_channel * self.TVG_residual_poly

        result = SssSimResult()
        result.sim_sss = sss_msg
        self.as_ping.set_succeeded(result)

if __name__ == '__main__':

    rospy.init_node('auv_sss_model', disable_signals=False)
    try:
        sss_model()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch mbes_model")
        pass

