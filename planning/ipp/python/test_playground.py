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

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models.transforms.outcome import Standardize
from matplotlib.animation import FuncAnimation
import filelock


current_pose = [0, 0, 0]

bounds_theta_torch = torch.tensor([[(current_pose[2]-np.pi/2)%(2*np.pi)],  [(current_pose[2] + np.pi/2)%(2*np.pi)]]).to(torch.float)



suggested_pose = [50, 0, np.pi]
t1 = time.time()
path = dubins.shortest_path(current_pose, suggested_pose, 9)
wp, length_arr = path.sample_many(1) 
cost = length_arr[-1] + 1

def get_orthogonal_samples(poses, nbr_samples=6, swath_width=18.0):
        """ Generates points on lines orthogonal to a vector. Will generate
            `nbr_samples` for each given vector, along a line of given swath width.
        

        Args:
            poses (list[float]): [x y theta]
            nbr_samples (int, optional): number of samples generated for each vector. Defaults to 6.
            swath_width (float, optional): width of line sampled from, for each vector. Defaults to 5.0.

        Returns:
            torch.Tensor: concatenated xy points of samples
        """
        radius = 0.5*swath_width
        samples = []
        for pose in poses:
            x = pose[0]
            y = pose[1]
            yaw = pose[2]

            dx = radius*np.sin(yaw) # shifted by 90 degree for orthogonality
            dy = radius*np.cos(yaw) 

            for i in np.linspace(-1, 1, 10):
                dx_s = dx * i
                dy_s = dy * i
                n1 = [x + dx_s, y - dy_s]
                n2 = [x - dx_s, y + dy_s]
                #all_samples.append(np.array([n1, n2]))
                samples.append(n1)
                samples.append(n2) 

        return samples
    
    

samples = get_orthogonal_samples(wp)
print(time.time()- t1)
print(len(samples)/cost)


for points in wp:
    x = points[0]
    y = points[1]

    plt.scatter(x, y, color=[0, 0, 1])

for points in samples:
    x = points[0]
    y = points[1]
    plt.scatter(x, y, color=[1, 0, 0])

plt.show()
    
t1 = time.time()
pcl = np.array(samples)

b = np.zeros((pcl.shape[0], pcl.shape[1] + 1))

b[:,:-1] = pcl


pcd3 = o3d.geometry.PointCloud()
pcd3.points = o3d.utility.Vector3dVector(b)

pcd3 = pcd3.voxel_down_sample(voxel_size=3)

#o3d.visualization.draw_geometries([pcd3])
xyz = np.asarray(pcd3.points)

xy = xyz[:, :2]

print(time.time()- t1)

print(len(xy)/cost)

ax = plt.gca()
plt.scatter(xy[:,0], xy[:,1])
ax.set_aspect('equal')
plt.show()


"""

n = 50

model2 = pickle.load(open(r"/home/alex/.ros/GP_angle.pickle","rb"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
model2.eval()
model2.likelihood.eval()
likelihood2 = GaussianLikelihood()
likelihood2.to(device).float()
model2.to(device).float()
torch.cuda.empty_cache()

samples1D = np.linspace(-np.pi, np.pi, n)

ucb_fun = UpperConfidenceBound(model2, 10)

mean_list = []
var_list = []
ucb_list = []

with torch.no_grad():
    inputst_temp = torch.from_numpy(samples1D).to(device).float()
    outputs = model2(inputst_temp)
    mean_r, sigma_r = ucb_fun._mean_and_sigma(inputst_temp)
    ucb = mean_r + ucb_fun.beta.sqrt() * sigma_r
    outputs = likelihood2(outputs)
    mean_list.append(outputs.mean.cpu().numpy())
    var_list.append(outputs.variance.cpu().numpy())
    ucb_list.append(ucb.cpu().numpy())


mean = np.vstack(mean_list).squeeze(0)
variance = np.vstack(var_list).squeeze(0)
ucb = np.vstack(ucb_list)

print(mean)
print(variance)

plt.plot(samples1D, mean)
plt.fill_between(samples1D, mean+variance, mean-variance, facecolor='blue', alpha=0.5)
plt.show()
"""
"""
# Reconstruct model
model = pickle.load(open(r"/home/alex/.ros/Wed, 17 Apr 2024 15:39:46_iteration_2405_GP.pickle","rb"))
model.model.eval()
model.likelihood.eval()
likelihood = GaussianLikelihood()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
likelihood.to(device).float()
model.to(device).float()

print(model.model.mean_module.constant.item())

points = model.variational_strategy.inducing_points.detach().numpy()
torch.cuda.empty_cache()

n = 200
n_contours = 50

# posterior sampling locations
inputsg = [
    np.linspace(min(points[:, 0])-50, max(points[:, 0])+50, n),
    np.linspace(min(points[:, 1])-50, max(points[:, 1])+50, n)
]
inputst = np.meshgrid(*inputsg)
s = inputst[0].shape
inputst = [_.flatten() for _ in inputst]
inputst = np.vstack(inputst).transpose()

ucb_fun = UpperConfidenceBound(model, 10)

mean_list = []
var_list = []
ucb_list = []
divs = 10
with torch.no_grad():
    for i in range(0, divs):
        # sample
        inputst_temp = torch.from_numpy(inputst[i*int(n*n/divs):(i+1)*int(n*n/divs), :]).to(device).float()
        outputs = model(inputst_temp)
        mean_r, sigma_r = ucb_fun._mean_and_sigma(inputst_temp)
        ucb = abs(mean_r - model.model.mean_module.constant) + ucb_fun.beta.sqrt() * sigma_r
        outputs = likelihood(outputs)
        mean_list.append(outputs.mean.cpu().numpy())
        var_list.append(outputs.variance.cpu().numpy())
        ucb_list.append(ucb.cpu().numpy())

mean = np.vstack(mean_list).reshape(s)
variance = np.vstack(var_list).reshape(s)
ucb = np.vstack(ucb_list).reshape(s)

# plot raw, mean, and variance
#levels = np.linspace(min(targets), max(targets), n_contours)
fig, ax = plt.subplots(3, sharex=True, sharey=True)
#cr = ax[0].scatter(inputs[:, 0], inputs[:, 1], c=targets,
#                    cmap='jet', s=0.4, edgecolors='none')
cm = ax[0].contourf(*inputsg, mean, cmap='jet', levels=n_contours)  # Normalized across plots
# cm = ax[1].contourf(*inputsg, mean, cmap='jet', levels=n_contours)
cv = ax[1].contourf(*inputsg, variance, levels=n_contours)
indpts = model.variational_strategy.inducing_points.data.cpu().numpy()
ax[1].plot(indpts[:, 0], indpts[:, 1], 'ko', markersize=1, alpha=0.5)
ca = ax[2].contourf(*inputsg, ucb, levels=n_contours)

post_cloud = np.hstack((inputst, mean.reshape(-1, 1)))
#np.save("./posterior.npy", post_cloud)

# colorbars
#fig.colorbar(cr, ax=ax[0])
fig.colorbar(cm, ax=ax[0])
fig.colorbar(cv, ax=ax[1])
fig.colorbar(ca, ax=ax[2])

# # formatting
#ax[0].set_aspect('equal')
#ax[0].set_title('Raw data')
#ax[0].set_ylabel('$y~[m]$')

ax[0].set_aspect('equal')
ax[0].set_title('Mean')
ax[0].set_ylabel('$y~[m]$')
ax[1].set_aspect('equal')
ax[1].set_title('Variance')
ax[1].set_ylabel('$y~[m]$')
ax[2].set_aspect('equal')
ax[2].set_title('UCB')
ax[2].set_ylabel('$y~[m]$')
ax[2].set_xlabel('$x~[m]$')
plt.tight_layout()
plt.show()

# Plot particle trajectory
#ax[0].plot(track[:,0], track[:,1], "-r", linewidth=0.2)

# # save
#fig.savefig(fname, bbox_inches='tight', dpi=1000)

# Free up GPU mem
del inputst
torch.cuda.empty_cache()


"""

"""

model = pickle.load(open(r"/home/alex/.ros/Wed, 17 Apr 2024 09:30:01_iteration_833_GP.pickle","rb"))


model.model.eval()
model.likelihood.eval()

n = 500
test_x = torch.zeros(int(pow(n, 2)), 2)
x_min = -250
x_max = 50
y_min = -150
y_max = 150
for i, x in enumerate(np.linspace(x_min, x_max, n)):
    for j, y in enumerate(np.linspace(y_min, y_max, n)):
        test_x[n*i + j][0] = x
        test_x[n*i + j][1] = y

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = model.likelihood(model.model(test_x))

pred_labels = observed_pred.mean.view(n,n)


# Calc abosolute error
#test_y_actual = torch.sin(((test_x[:, 0] + test_x[:, 1]) * (2 * np.pi))).view(n, n)
#delta_y = torch.abs(pred_labels - test_y_actual).detach().numpy()

# Define a plotting function
def ax_plot(f, ax, y_labels, title):
    im = ax.imshow(y_labels)
    ax.set_title(title)
    f.colorbar(im)

# Plot our predictive means
cs = plt.contourf(pred_labels, cmap='jet')
  
cbar = plt.colorbar(cs) 
#plt.imshow(pred_labels, extent=[x_min, x_max, y_min, y_max], origin="lower")
points = model.variational_strategy.inducing_points.detach().numpy()
plt.scatter(points[:,0], points[:,1])
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










"""

train_X = torch.rand(10, 2, dtype=torch.float64)
# explicit output dimension -- Y is 10 x 1
train_Y = 1 - (train_X - 0.5)
train_Y = train_Y.norm(dim=-1, keepdim=True)
train_Y += 0.1 * torch.rand_like(train_Y)

print(train_X)
print(train_X.dtype)
print(train_X.grad_fn)
print(train_Y)
print(train_Y.dtype)
print(train_Y.grad_fn)

gp = SingleTaskGP(train_X, train_Y, outcome_transform=Standardize(m=1))
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)
"""



"""
x = torch.from_numpy(np.random.uniform(low=[0, 0, -np.pi], 
                               high=[10, 10, np.pi], size=[10, 3]))
print(x)
"""
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
    r = 1
    for pose in poses:
        x = pose[0]
        y = pose[1]
        yaw = pose[2]

        dx = r/2*np.sin(yaw) # shifted by 90 degree for orthogonality
        dy = r/2*np.cos(yaw) 

        for i in np.linspace(-1, 1, 5):
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