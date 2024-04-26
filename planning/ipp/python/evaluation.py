import numpy as np
import scipy
from scipy.spatial.distance import cdist
import open3d as o3d

# STEP 1: Get the underlying mesh used

# STEP 2: Get ndarray of underlying mesh

# STEP 3: 


pcl = np.load(r"/home/alex/catkin_ws/src/UWExploration/utils/uw_tests/datasets/lost_targets/mesh.npz")

print(pcl.files)

print(pcl['V'])
print(pcl['F'])
print(pcl['bounds'])

#pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(pcl)

#print(pcd)
#pcd.estimate_normals()

"""
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9)
    
mesh.vertex_colors = o3d.utility.Vector3dVector(pcd.colors)
print(mesh)
o3d.visualization.draw_geometries([mesh])

o3d.visualization.draw_geometries([pcd])

"""

def _construct_query_tree(min_x: float, min_y: float, num_rows: float, num_cols: float, resolution: float):
    """_summary_

    Args:
        min_x (float): _description_
        min_y (float): _description_
        num_rows (float): _description_
        num_cols (float): _description_
        resolution (float): _description_

    Returns:
        _type_: _description_
    """

    x = np.linspace(min_x + (0.5 * resolution), min_x + (num_rows - 1 + 0.5) * resolution, num_rows)
    y = np.linspace(min_y + (0.5 * resolution), min_y + (num_cols - 1 + 0.5) * resolution, num_cols)

    xv, yv = np.meshgrid(x, y)
    queries = np.stack((xv.flatten(), yv.flatten()), axis=-1)
    queries_tree = scipy.spatial.KDTree(queries)

    return queries, queries_tree

