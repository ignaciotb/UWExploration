#!/usr/bin/env python3
from auvlib.data_tools import std_data, gsf_data, xyz_data, csv_data, all_data
from auvlib.bathy_maps import draw_map, mesh_map
import sys
import numpy as np
import open3d as o3d
from decimal import *
from scipy.spatial.transform import Rotation as rot
import configargparse
import gc

getcontext().prec = 28
p = configargparse.ArgumentParser()
p.add('-c', '--config_filepath', required=False, is_config_file=True, help='Path to config file.')
p.add_argument('--mbes_type', type=str, default="all")
p.add_argument('--mbes_path', type=str, default="/media/nacho/Seagate Expansion Drive/MMTDatasets/LostTargets/Ping_Processed/5-Products/MBES_GSF/Post_Deployment_PROCESSED/bathy")
p.add_argument('--svp_path', type=str, default="/media/torroba/Seagate Expansion Drive/MMTDatasets/LostTargets/SVP/csv")
opt = p.parse_args()


# Read MBES and save to cereal
if opt.mbes_type == "gsf":
    gsf_pings = gsf_data.gsf_mbes_ping.parse_folder(opt.mbes_path) 
    mbes_pings = gsf_data.convert_pings(gsf_pings) 
    del gsf_pings

elif opt.mbes_type == "all":
    all_pings = all_data.all_mbes_ping.parse_folder(opt.mbes_path)
    navi_entries = all_data.all_nav_entry.parse_folder(opt.mbes_path)
    mbes_pings = all_data.convert_matched_entries(all_pings, navi_entries)
    del navi_entries
    del all_pings

gc.collect()

# Removing first 10 pings for overnight dataset
#std_data.write_data(mbes_pings[10:], "mbes_pings.cereal")
std_data.write_data(mbes_pings, "mbes_pings.cereal")
print("cereal saved")

# Draw the height map
d = draw_map.BathyMapImage(mbes_pings, 500, 500) 
d.draw_height_map(mbes_pings) 
#d.draw_track(mbes_pings)
#d.show()
d.write_image("default_real_mean_depth.png") 
print("Image saved")

# Transform beams to avoid losing precision
R = rot.from_euler("zyx", [mbes_pings[0].heading_, mbes_pings[0].pitch_, 0.]).as_matrix()
pos = mbes_pings[0].pos_
R_inv = R.transpose()

# Create XYZ cloud from pings
cloud_xyz = xyz_data.cloud.from_pings(mbes_pings)
mbes_pings = None # Delete to save mem
cloud_np = np.concatenate(cloud_xyz, axis=0 )

# Downsample and save point cloud as NPY from transformed pings
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(cloud_np)
del cloud_np
gc.collect()
pcd = pcd.voxel_down_sample(voxel_size=0.5)
o3d.visualization.draw_geometries([pcd])
cloud_np = np.asarray(pcd.points)
del pcd
gc.collect()

pos_inv = R_inv.dot(pos)
cloud_np[:] = [R_inv.dot(p) - pos_inv for p in cloud_np]
np.save("pcl.npy", cloud_np)
print("Point cloud saved")

# Create and save mesh from transformed beams
mesh_res = 2
V, F, bounds = mesh_map.mesh_from_cloud(cloud_np, mesh_res)
del cloud_np
gc.collect()

#V, F, bounds = mesh_map.mesh_from_pings(mbes_pings, mesh_res)
np.savez("mesh.npz", V=V, F=F, bounds=bounds)

# To visualize the mesh
mesh_map.show_mesh(V,F)
del V, F, bounds 
gc.collect()

# Read SVP in CVS format and save to cereal
svp = csv_data.csv_asvp_sound_speed.parse_folder(opt.svp_path)
csv_data.write_data(svp, "svp.cereal")
