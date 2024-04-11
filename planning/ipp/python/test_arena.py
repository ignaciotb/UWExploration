import itertools
import warnings

import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models.approximate_gp import (
    _SingleTaskVariationalGP,
    ApproximateGPyTorchModel,
    SingleTaskVariationalGP,
)
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Log
from botorch.models.utils.inducing_point_allocators import (
    GreedyImprovementReduction,
    GreedyVarianceReduction,
)
from botorch.posteriors import GPyTorchPosterior, TransformedPosterior
from botorch.utils.testing import BotorchTestCase
from gpytorch.likelihoods import GaussianLikelihood, MultitaskGaussianLikelihood
from gpytorch.mlls import VariationalELBO
from gpytorch.variational import (
    IndependentMultitaskVariationalStrategy,
    VariationalStrategy,
)

import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import dubins

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

import numpy as np
import time
import random
import open3d as o3d
import pickle


current_pose = [0, 0, 0]
suggested_pose = [20, 15, 0]
d1= [suggested_pose[0] - current_pose[0], suggested_pose[1] - current_pose[1]]
n1 = np.sqrt(d1[0] ** 2 + d1[1] ** 2)
turning_radius = 3
wp_resolution = n1 / 50
print(n1)
print(wp_resolution)

path = dubins.shortest_path(current_pose, suggested_pose, turning_radius)
wp, length_arr = path.sample_many(wp_resolution) 
cost = length_arr[-1] + wp_resolution

t1 = time.time()

def get_orthogonal_samples(poses):
    all_samples = []
    for pose in poses:
        x = pose[0]
        y = pose[1]
        yaw = pose[2]

        dx = np.sin(yaw) # shifted by 90 degree for orthogonality
        dy = np.cos(yaw) 

        for i in np.linspace(0.2, 1, 3):
            dx_s = dx * i
            dy_s = dy * i
            n1 = [x + dx_s, y - dy_s]
            n2 = [x - dx_s, y + dy_s]
            all_samples.append(np.array([n1, n2]))

        #n1 = [x + dx, y - dy]
        #n2 = [x - dx, y + dy]
        #orthogonal_samples = np.random.uniform(low=[min(x + dx, x - dx), min(y - dy, y + dy)], high=[max(x + dx, x - dx), max(y - dy, y + dy)], size=[10, 2])
        #all_samples.append(np.array([n1, n2]))
    return all_samples


wp_samples = wp[::5]

point_list = get_orthogonal_samples(wp_samples)

print(time.time()-t1)

plt.figure(figsize = (10,8))
for points in point_list:
    plt.scatter(points[:, 0], points[:, 1])

                    
model = pickle.load(open(r"/home/alex/.ros/Thu, 11 Apr 2024 18:58:57_iteration_930_GP.pickle","rb"))

#print(wp)
#print(cost)

x_val = [x[0] for x in wp]
y_val = [x[1] for x in wp]
yaw_val = [x[2] for x in wp]

#print(x_val, y_val)

plt.plot(x_val, y_val)
plt.axis('scaled')
plt.show()







"""
corners = [-250, -150, 0, -60]

n_ip = 10
minibatch = 5
lr = 0.01




costs = torch.Tensor()

a = torch.cat((costs,torch.Tensor([0.1])),0)

print(costs)

print(a)
"""


"""
bounds = torch.tensor([[corners[0], corners[3]], [corners[1], corners[2]]]).to(torch.float)
print(bounds)
initial_x = torch.randn(4,2)
print(initial_x)
var_dist = CholeskyVariationalDistribution(5)
model = SingleTaskVariationalGP(
    train_X=initial_x,
    num_outputs=1,
    variational_distribution=var_dist,
    inducing_points = n_ip,
    learn_inducing_points=True,
    mean_module=gpytorch.means.ConstantMean(),
    covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))
model.variational_strategy = VariationalStrategy(model, torch.randn((5, 1)), var_dist, learn_inducing_locations=True)

likelihood = GaussianLikelihood()
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mll = gpytorch.mlls.VariationalELBO(likelihood, model.model, minibatch, combine_terms=True)
likelihood.to(device).float()
model.to(device).float()

opt = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=float(lr))


aq_func = UpperConfidenceBound(model, beta=0.1)

candidate, acq_value = optimize_acqf(aq_func, bounds=bounds, q=1, num_restarts=5, raw_samples=20)
"""