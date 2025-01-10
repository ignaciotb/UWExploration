#!/usr/bin/env python3

# Standard dependencies
import math
import rospy
import sys
import numpy as np
import time

# from scipy.spatial.transform import Rotation as rot

from tf.transformations import quaternion_matrix, euler_from_quaternion
from tf.transformations import rotation_matrix

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from scipy.ndimage import gaussian_filter1d

# For sim mbes action client
import actionlib
from auv_model.msg import MbesSimAction, MbesSimResult
from auv_model.msg import SssSimAction, SssSimResult, Sidescan

# Auvlib
from auvlib.bathy_maps import base_draper
from auvlib.data_tools import csv_data, xtf_data

class auv_payload(object):

    def __init__(self):

        # Load mesh
        svp_path = rospy.get_param('~sound_velocity_prof')
        mesh_path = rospy.get_param('~mesh_path')
        self.mbes_angle = rospy.get_param("~mbes_open_angle", np.pi/180. * 60.)
        self.mbes_frame = rospy.get_param(
            '~mbes_link', 'mbes_link')  # mbes frame_id

        if svp_path.split('.')[1] != 'cereal':
            sound_speeds = csv_data.csv_asvp_sound_speed.parse_file(svp_path)
        else:
            sound_speeds = csv_data.csv_asvp_sound_speed.read_data(svp_path)

        data = np.load(mesh_path)
        V, F, bounds = data['V'], data['F'], data['bounds']
        print(bounds)
        print("Mesh loaded")

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
        self.avg_time = []

        # Create draper
        self.draper = base_draper.BaseDraper(V, F, bounds, isovelocity_sound_speeds)
        # self.draper.set_ray_tracing_enabled(False)
        self.draper.set_sidescan_yaw(0)
        sensor_offset =  np.array([0.0,0.0,0.])
        self.draper.set_sidescan_port_stbd_offsets(sensor_offset, sensor_offset)

        data = None
        V = None
        F = None
        bounds = None
        sound_speeds = None
        print("draper created")
        print("Size of draper: ", sys.getsizeof(self.draper))

        # Action server for MBES pings sim (necessary to be able to use UFO maps as well)
        sim_mbes_as = rospy.get_param('~mbes_sim_as', '/mbes_sim_server')
        server_mode = rospy.get_param("~server_mode", False)
        self.as_mbes = actionlib.SimpleActionServer(sim_mbes_as, MbesSimAction,
                                                    execute_cb=self.mbes_as_cb, auto_start=server_mode)
        
        # Action server for S pings sim (necessary to be able to use UFO maps as well)
        sim_sss_as = rospy.get_param('~sss_sim_as', '/sss_sim_server')
        server_mode = rospy.get_param("~server_mode", False)
        self.as_ping = actionlib.SimpleActionServer(sim_sss_as, SssSimAction,
                                                    execute_cb=self.sss_as_cb, auto_start=server_mode)

        rospy.spin()


    # Action server to simulate MBES for the sim AUV
    def mbes_as_cb(self, goal):

        # Unpack goal
        p_mbes = [goal.mbes_pose.transform.translation.x,
                  goal.mbes_pose.transform.translation.y,
                  goal.mbes_pose.transform.translation.z]
        r_mbes = quaternion_matrix([goal.mbes_pose.transform.rotation.x,
                                    goal.mbes_pose.transform.rotation.y,
                                    goal.mbes_pose.transform.rotation.z,
                                    goal.mbes_pose.transform.rotation.w])[0:3, 0:3]

        # IGL sim ping
        # The sensor frame on IGL needs to have the z axis pointing opposite from the actual sensor direction
        R_flip = rotation_matrix(np.pi, (1, 0, 0))[0:3, 0:3]
        mbes = self.draper.project_mbes(np.asarray(p_mbes), r_mbes,
                                        goal.beams_num.data, self.mbes_angle)

        mbes = mbes[::-1]  # Reverse beams for same order as real pings

        # Transform points to MBES frame (same frame than real pings)
        rot_inv = r_mbes.transpose()
        p_inv = rot_inv.dot(p_mbes)
        mbes = np.dot(rot_inv, mbes.T)
        mbes = np.subtract(mbes.T, p_inv)

        # Add noise
        # mbes = gaussian_filter1d(mbes , sigma=0.5)


        # Pack result
        mbes_cloud = self.pack_cloud(self.mbes_frame, mbes)
        result = MbesSimResult()
        result.sim_mbes = mbes_cloud
        self.as_mbes.set_succeeded(result)


    # Create PointCloud2 msg out of ping
    def pack_cloud(self, frame, mbes):
        mbes_pcloud = PointCloud2()
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1)]

        mbes_pcloud = point_cloud2.create_cloud(header, fields, mbes)

        return mbes_pcloud
    
    # Action server to simulate SSS for the sim AUV
    def sss_as_cb(self, goal):

        # Unpack goal
        p_sss = [goal.sss_pose.transform.translation.x,
                  goal.sss_pose.transform.translation.y,
                  goal.sss_pose.transform.translation.z]
        euler_sss = euler_from_quaternion([goal.sss_pose.transform.rotation.x,
                                            goal.sss_pose.transform.rotation.y,
                                            goal.sss_pose.transform.rotation.z,
                                            goal.sss_pose.transform.rotation.w])

        # Create ping to render SSS
        xtf_ping = xtf_data.xtf_sss_ping()
        xtf_ping.port.time_duration = self.max_r*2/self.svp
        xtf_ping.stbd.time_duration = self.max_r*2/self.svp
        xtf_ping.pos_ = np.array([p_sss[0],p_sss[1],p_sss[2]])
        xtf_ping.roll_, xtf_ping.pitch_, xtf_ping.heading_ = euler_sss

        start_time = time.time()
        # Rendering
        nbr_bins = goal.beams_num.data
        print("Num of beams ", nbr_bins)
        left, right = self.draper.project_ping(xtf_ping, nbr_bins) # project
        # sss = sss[::-1]  # Reverse beams for same order as real pings
        
        self.avg_time.append(time.time() - start_time)
        if len(self.avg_time) == 10:
            print("Raytracing time ", (np.asarray(self.avg_time).sum()/10.))
            self.avg_time.pop()

        port_channel = np.copy(np.array(left.time_bin_model_intensities)*255.)
        starboard_channel = np.copy(np.array(right.time_bin_model_intensities)*255.)
        port_channel[np.isnan(port_channel)] = 0
        starboard_channel[np.isnan(starboard_channel)] = 0

        sss_msg = Sidescan()
        sss_msg.port_channel = port_channel.astype(int).tolist()
        sss_msg.starboard_channel = starboard_channel.astype(int).tolist()

        ## If we want to apply TVG residual
        # slant_range = np.arange(nbr_bins)*self.max_r/nbr_bins
        # theta_port = np.ones_like(slant_range)*np.pi/2
        # theta_stbd = -np.ones_like(slant_range)*np.pi/2

        # # for TVG residual (iSSS)
        # ab_poly =  [ 2.24377726e-04, -5.62777122e-03,  5.51645115e-02,  4.02042166e-01]
        # self.TVG_residual_poly = np.polyval(ab_poly, slant_range)
        # self.TVG_residual_poly[self.TVG_residual_poly>4]=4
        # sss_msg.port_channel = sss_msg.port_channel * self.TVG_residual_poly
        # sss_msg.starboard_channel = sss_msg.starboard_channel * self.TVG_residual_poly

        result = SssSimResult()
        result.sim_sss = sss_msg
        self.as_ping.set_succeeded(result)


if __name__ == '__main__':

    rospy.init_node('auv_payload', disable_signals=False)
    try:
        auv_payload()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch auv_payload")
        pass

