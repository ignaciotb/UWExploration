#!/usr/bin/env python

import numpy as np
from auvlib.data_tools import gsf_data, std_data, csv_data, xyz_data
from auvlib.bathy_maps import mesh_map, base_draper
import configargparse
import math
import os
import sys
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import stats
from scipy.spatial.transform import Rotation as rot
from decimal import *

p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--mbes_path', type=str, default='/home/torroba18/Downloads/post_deployment/mbes_pings.cereal')
p.add_argument('--svp_path', type=str, default='/home/torroba18/Downloads/post_deployment/KTH_PI_SVP_20180807_1251_573365N_0115014E_004.asvp')
p.add_argument('--cloud_path', type=str, default='/home/torroba18/Downloads/post_deployment/KTH_Post_Deployment_AVG_WGS84UTM32N_RH200_50cm.xyz',
               help='Options are "sine" (all sine activations) and "mixed" (first layer sine, other layers tanh)')
opt = p.parse_args()

mesh_res = 0.6
beam_width = math.pi/180.*60.

cloud = xyz_data.cloud.parse_file(opt.cloud_path)
pings = std_data.mbes_ping.read_data(opt.mbes_path)

getcontext().prec = 28

R = rot.from_euler("zyx", [pings[0].heading_, pings[0].pitch_, 0.]).as_dcm()
pos = pings[0].pos_
R_inv = R.transpose()

pos_inv = R_inv.dot(pos)
cloud[:] = [R_inv.dot(p) - pos_inv for p in cloud]

#  V, F, bounds = mesh_map.mesh_from_cloud(cloud, mesh_res)
V, F, bounds = mesh_map.mesh_from_dtm_cloud(cloud, mesh_res)
np.savez("mesh.npz", V=V, F=F, bounds=bounds)

mesh_map.show_mesh(V,F)
