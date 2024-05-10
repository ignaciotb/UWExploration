import numpy as np
import scipy
from scipy.spatial.distance import cdist
import open3d as o3d
import pickle
import torch
import gpytorch
import matplotlib.pyplot as plt
from consistency_plot import *

# STEP 1: Get the underlying mesh used

# STEP 2: Get ndarray of underlying mesh

# STEP 3: 

gp = pickle.load(open(r"/home/alex/.ros/GP_env.pickle","rb"))
gp.model.eval()
gp.likelihood.eval()
likelihood1 = gpytorch.likelihoods.GaussianLikelihood()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
likelihood1.to(device).float()
gp.to(device).float()
torch.cuda.empty_cache()

resolution = 0.5

gp2 = pickle.load(open(r"/home/alex/.ros/GP_env_lawnmower.pickle","rb"))
gp2.model.eval()
gp2.likelihood.eval()
likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
likelihood2.to(device).float()
gp2.to(device).float()
torch.cuda.empty_cache()

bounds = [592, 821, -179, -457]

# posterior sampling locations for first GP
inputsg = [
    np.arange(bounds[0], bounds[1], resolution),
    np.arange(bounds[3], bounds[2], resolution)
]
inputst = np.meshgrid(*inputsg)
s = inputst[0].shape
inputst = [_.flatten() for _ in inputst]
inputst = np.vstack(inputst).transpose()

mean_list = []
divs = 10
with torch.no_grad():
    # sample
    inputst_temp = torch.from_numpy(inputst).to(device).float()
    outputs = gp(inputst_temp)
    outputs = likelihood1(outputs)
    mean_list.append(outputs.mean.cpu().numpy())

mean2_list = []
with torch.no_grad():
    # sample
    inputst_temp = torch.from_numpy(inputst).to(device).float()
    outputs = gp2(inputst_temp)
    outputs = likelihood2(outputs)
    mean2_list.append(outputs.mean.cpu().numpy())

mean2 = np.squeeze(np.array(mean2_list), 0)
mean2 = np.expand_dims(mean2, 1)

mean = np.squeeze(np.array(mean_list), 0)
mean = np.expand_dims(mean, 1)

a = np.concatenate((inputst, mean), 1)
b = np.concatenate((inputst, mean2), 1)



#plt.contourf(*inputsg, mean, cmap='jet', levels=50)
#print(mean)
#plt.show()
#mean = np.expand_dims(mean, 0)
#b = np.meshgrid(*inputsg)
#print(np.shape(mean))
#print(np.shape(b))

#c = np.concatenate((b, mean), axis=0)
#print(np.shape(c))
#print(c)
#d = np.reshape(c, (170*220, 3))
#
#print(np.shape(d))

#print(d)

#print(c)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(a)

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(b)

pcl3 = np.load(r"/home/alex/catkin_ws/src/UWExploration/utils/uw_tests/datasets/asko/pcl.npy")
pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(pcl3)

pcd3 = pcd3.voxel_down_sample(voxel_size=1)

pcd3, _ = pcd3.remove_statistical_outlier(100, 0.5)

#o3d.visualization.draw_geometries([pcd])
o3d.visualization.draw_geometries([pcd3])
#o3d.visualization.draw_geometries([pcd, pcd3])

source = np.asarray(pcd.points)
ref = np.asarray(pcd3.points)

k = compute_consistency_metrics(source, ref, 2, True, True)


