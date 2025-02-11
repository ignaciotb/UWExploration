import numpy as np 
import open3d as o3d 
from auvlib.bathy_maps import draw_map, mesh_map 
import matplotlib.pyplot as plt
import cv2

def mesh_from_heightmap(img_path):

    bounds = []
    bounds.append([-326.959, -296.304])
    bounds.append([84.041, 48.696])
    bounds = np.array(bounds, dtype=np.float64)  
    # bounds = bounds.reshape([2,2]).tolist()
    print(bounds)

    img = plt.imread(img_path)
    plt.imshow(img)
    plt.axis('off')  # Optional: Turn off the axis for a cleaner display
    plt.show()
    if img.shape[-1] == 4:  # RGBA
        image_rgb = img[:, :, :3]  # Remove the alpha channel (keep RGB)
    else:
        image_rgb = img  # No alpha channel, keep the image as is

    V, F = mesh_map.mesh_from_height_map(image_rgb, bounds)
    mesh_map.show_mesh(V,F)
    np.savez("mesh_from_img.npz", V=V, F=F, bounds=bounds) 


def read_and_plot_point_cloud(npy_file): 
    # Load point cloud data from .npy file 
    point_cloud_data = np.load(npy_file) 

    # Check if the data is in the correct shape (Nx3 for points) 
    if point_cloud_data.shape[1] != 3:
        raise ValueError("The point cloud data must have three columns for X, Y, and Z coordinates.")
    
    # Create an Open3D PointCloud object 
    point_cloud = o3d.geometry.PointCloud() 

    # Assign the points to the PointCloud object 
    point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data) 
    # print(point_cloud.points[0]) 
    
    # Save as np and make relative to map
    cloud_np = np.asarray(point_cloud.points) 
    map_p = np.array([652118.482, 6523354.593, 0.000]) 
    cloud_np[:] = [p - map_p for p in cloud_np] 

    # # Build and save mesh
    mesh_res = 0.5 
    # heightmap, bounds = mesh_map.height_map_from_dtm_cloud(cloud_np, mesh_res)
    # print(bounds)
    # cv2.imwrite("heightmap.png", heightmap) 

    # plt.imshow(heightmap, cmap='gray')
    # plt.axis('off')  # Remove axis for cleaner output
    # plt.savefig("heightmap.png", bbox_inches='tight', pad_inches=0)
    
    # np.set_printoptions(threshold=np.inf)
    # # print(heightmap)
    # V, F = mesh_map.mesh_from_height_map(heightmap, bounds)
    # mesh_map.show_mesh(V,F)
    
    V, F, bounds = mesh_map.mesh_from_cloud(cloud_np, mesh_res) 
    mesh_map.show_mesh(V,F)
    print(bounds)
    np.set_printoptions(threshold=np.inf)
    print(V)
    
    np.savez("mesh.npz", V=V, F=F, bounds=bounds) 
    np.save('vertex.npy', V)
    np.save('edges.npy', F)
    np.save('bounds.npy', bounds)
    V = None
    F = None
    cloud_np = None

    # Apply a voxel grid filter to original pointcloud
    voxel_size = 1.  # Set the voxel size 
    voxel_downsampled = point_cloud.voxel_down_sample(voxel_size) 
    voxel_np = np.asarray(voxel_downsampled.points) 
    voxel_np[:] = [p - map_p for p in voxel_np] 
    np.save("pcl.npy", voxel_np) 

    # Visualize the voxel-filtered point cloud 
    o3d.visualization.draw_geometries([voxel_downsampled], 
                                    window_name="Voxel Grid Filtered Point Cloud Viewer", 
                                    width=800, height=600, 
                                    left=50, top=50, 
                                    point_show_normal=False) 
  
# Example usage 
# Replace 'your_point_cloud.npy' with the path to your .npy file 
npy_file_path = "/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/utils/uw_tests/datasets/asko/Asko_xyz.npy" 
img_file_path = "/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/utils/uw_tests/datasets/asko/asko_bay_dv/heightmap.png" 
read_and_plot_point_cloud(npy_file_path) 
# mesh_from_heightmap(img_file_path)

