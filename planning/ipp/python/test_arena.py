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

model = pickle.load(open(r"/home/alex/.ros/Mon, 15 Apr 2024 08:26:50_iteration_1069_GP.pickle","rb"))


model.model.eval()
model.likelihood.eval()
n = 100
test_x = torch.zeros(int(pow(n, 2)), 2)
x_min = -250
x_max = -150
y_min = 0
y_max = -60
for i, x in enumerate(np.linspace(x_min, x_max, n)):
    for j, y in enumerate(np.linspace(y_min, y_max, n)):
        test_x[n*i + j][0] = x 
        test_x[n*i + j][1] = y

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = model.likelihood(model.model(test_x))

pred_labels = observed_pred.mean.view(n,n)

print(pred_labels)

# Calc abosolute error
#test_y_actual = torch.sin(((test_x[:, 0] + test_x[:, 1]) * (2 * np.pi))).view(n, n)
#delta_y = torch.abs(pred_labels - test_y_actual).detach().numpy()

# Define a plotting function
def ax_plot(f, ax, y_labels, title):
    im = ax.imshow(y_labels)
    ax.set_title(title)
    f.colorbar(im)

# Plot our predictive means
plt.imshow(pred_labels, extent=[x_min, x_max, y_min, y_max])
points = model.variational_strategy.inducing_points.detach().numpy()
plt.scatter(points[:,0], points[:,1])
plt.colorbar()
plt.show()


#f, observed_ax = plt.subplots(1, 1, figsize=(4, 3))
#ax_plot(f, observed_ax, pred_labels, 'Predicted Mean Values')

# Plot the true values
#f, observed_ax2 = plt.subplots(1, 1, figsize=(4, 3))
#ax_plot(f, observed_ax2, test_y_actual, 'Actual Values')

# Plot the absolute errors
#f, observed_ax3 = plt.subplots(1, 1, figsize=(4, 3))
#ax_plot(f, observed_ax3, delta_y, 'Absolute Error Surface')


"""
model = pickle.load(open(r"/home/alex/.ros/Thu, 11 Apr 2024 18:58:57_iteration_930_GP.pickle","rb"))

x = model.variational_strategy.get_fantasy_model()

print(x)

"""

"""
inputsg = [
        np.linspace(min(inputs[:, 0]), max(inputs[:, 0]), n),
        np.linspace(min(inputs[:, 1]), max(inputs[:, 1]), n)
    ]
inputst = np.meshgrid(*inputsg)
s = inputst[0].shape
inputst = [_.flatten() for _ in inputst]
inputst = np.vstack(inputst).transpose()

mean_list = []
var_list = []
divs = 10
with torch.no_grad():
    for i in range(0, divs):
        # sample
        inputst_temp = torch.from_numpy(inputst[i*int(n*n/divs):(i+1)*int(n*n/divs), :]).to(device).float()
        outputs = model(inputst_temp)
        outputs = likelihood(outputs)
        mean_list.append(outputs.mean.cpu().numpy())
        var_list.append(outputs.variance.cpu().numpy())

mean = np.vstack(mean_list).reshape(s)
variance = np.vstack(var_list).reshape(s)



# plot raw, mean, and variance
levels = np.linspace(min(targets), max(targets), n_contours)
fig, ax = plt.subplots(3, sharex=True, sharey=True)
cr = ax[0].scatter(inputs[:, 0], inputs[:, 1], c=targets,
                    cmap='jet', s=0.4, edgecolors='none')
cm = ax[1].contourf(*inputsg, mean, cmap='jet', levels=levels)  # Normalized across plots
# cm = ax[1].contourf(*inputsg, mean, cmap='jet', levels=n_contours)
cv = ax[2].contourf(*inputsg, variance, levels=n_contours)
indpts = model.variational_strategy.inducing_points.data.cpu().numpy()
ax[2].plot(indpts[:, 0], indpts[:, 1], 'ko', markersize=1, alpha=0.2)

"""

"""
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