# Auvlib
from auvlib.bathy_maps import base_draper, mesh_map
from auvlib.data_tools import csv_data, xyz_data, std_data, xtf_data

import numpy as np
import matplotlib.pyplot as plt 

from smarc_msgs.msg import Sidescan


# - For Dec 2024, Asko 
# [estimated using logged temperature from DVL from Dec and measured SVPs from Oct 2024 (assuming salinity is the same)]
# 1431.1 m/s is assuming salinity doesn't change 
# 1430.1 m/s is eyeballing from the altitude (using mesh) and nadir line in the sidescan waterfall image

svp = 1431.1 - 1 # mean sound speed (m/s) 
isovelocity_sound_speeds = [csv_data.csv_asvp_sound_speed()]
isovelocity_sound_speeds[0].dbars = np.arange(50)
isovelocity_sound_speeds[0].vels = np.ones((50))*svp # m/s
max_r = 40 # this is the number we've put in the sss driver when running the mission
max_r = max_r/1500.0*svp # this is the actual max range accounting for the sound speed
nbr_bins = 1000
slant_range = np.arange(nbr_bins)*max_r/nbr_bins
theta_port = np.ones_like(slant_range)*np.pi/2
theta_stbd = -np.ones_like(slant_range)*np.pi/2

# for TVG residual (iSSS)
ab_poly =  [ 2.24377726e-04, -5.62777122e-03,  5.51645115e-02,  4.02042166e-01]
TVG_residual_poly = np.polyval(ab_poly, slant_range)
TVG_residual_poly[TVG_residual_poly>4]=4

clouds_path = "/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/utils/uw_tests/datasets/asko/Asko_xyz.npy"
xyz_clouds = np.load(clouds_path)
heightmap, bounds = mesh_map.height_map_from_dtm_cloud(xyz_clouds, 0.5)
auvlib_mbes_V, auvlib_mbes_F = mesh_map.mesh_from_height_map(heightmap,bounds)

draper = base_draper.BaseDraper(auvlib_mbes_V, auvlib_mbes_F, bounds, isovelocity_sound_speeds)
draper.set_sidescan_yaw(0)

# this is for iSSS stick
lever_arm = np.array([0.0,0.0,-1.6])

# this is because the orthometric height from RTK GPS uses a different geoid model than the one used in MBES
undulation_offset = np.array([0.0,0.0,1.4151990000000012])
sensor_offset =  lever_arm + undulation_offset
# print(f"appliying sensor offset: {sensor_offset}")
draper.set_sidescan_port_stbd_offsets(sensor_offset, sensor_offset)

# psudo code
nbr_particles = 10
nbr_sss_pings = 100

SSS_waterfall_image = np.zeros((nbr_sss_pings, 2*nbr_bins))
# SSS_model_waterfall_image = np.zeros((nbr_sss_pings, 2*nbr_bins))
PF = [{"SSS": np.zeros((nbr_sss_pings, 2*nbr_bins))} for i in range(nbr_particles)]

# placeholder for sss pings
sss_msgs = [Sidescan() for i in range(nbr_sss_pings)]

# placeholder for particle poses
particle_poses = np.zeros((nbr_sss_pings, nbr_particles, 6))

# get real sss waterfall image 
for ping_num in range(nbr_sss_pings):
    msg = sss_msgs[ping_num]
    port_amplitude = np.copy(np.array(bytearray(msg.port_channel), dtype=np.uint8))
    stbd_amplitude = np.copy(np.array(bytearray(msg.starboard_channel), dtype=np.uint8))
    SSS_waterfall_image[ping_num] = np.hstack((np.flip(stbd_amplitude), port_amplitude)) # 0-255

# I think we only need to intialize xtf_sss_ping once
xtf_ping = xtf_data.xtf_sss_ping()
xtf_ping.port.time_duration = max_r*2/svp
xtf_ping.stbd.time_duration = max_r*2/svp
    
for i in range(nbr_particles):
    for ping_num in range(nbr_sss_pings):

        # x,y,z in meters (ENU)
        # roll, pitch, yaw in radians (ENU)
        x,y,z,roll,pitch,yaw = particle_poses[ping_num][i] #[x,y,z,roll,pitch,yaw]

        xtf_ping.pos_ = np.array([x, y, z])
        xtf_ping.roll_ = roll
        xtf_ping.pitch_ = pitch
        xtf_ping.heading_ = yaw

        left, right = draper.project_ping(xtf_ping, nbr_bins) # project
        # this is based on Lambertian model with sonar equation
        # And you might want to comment out some lines in auvlib... I havn't tested this yet
        # https://github.com/nilsbore/auvlib/blob/dea44b02a51b752eb4b270da34e47a536f88b525/src/bathy_maps/src/base_draper.cpp#L325
        port_model_amplitude = np.copy(np.array(left.time_bin_model_intensities)*255.)
        stbd_model_amplitude = np.copy(np.array(right.time_bin_model_intensities)*255.)
        port_model_amplitude[np.isnan(port_model_amplitude)] = 0
        stbd_model_amplitude[np.isnan(stbd_model_amplitude)] = 0
        # if we want to apply TVG residual
        port_model_amplitude = port_model_amplitude * TVG_residual_poly
        stbd_model_amplitude = stbd_model_amplitude * TVG_residual_poly

        # SSS_model_waterfall_image[ping_num] = np.hstack((np.flip(stbd_model_amplitude), port_model_amplitude))
        PF[i]["SSS"][ping_num] = np.hstack((np.flip(stbd_model_amplitude), port_model_amplitude))
    
    # compute difference between model and real
    invalid_mask = PF[i]["SSS"]==0    
    diff = np.abs(SSS_waterfall_image - PF[i]["SSS"])
    error = diff[~invalid_mask].mean()

    #plot 
    fig, ax = plt.subplots(3,1,sharex=True, sharey=True)
    ax[0].imshow(SSS_waterfall_image, cmap='gray', vmin=0, vmax=255)
    ax[1].imshow(PF[i]["SSS"], cmap='gray', vmin=0, vmax=255)
    ax[2].imshow(diff, cmap='gray')
    plt.show()
    
    