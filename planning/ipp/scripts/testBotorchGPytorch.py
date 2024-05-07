#!/usr/bin/env python3

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
import tqdm
import gpytorch
from matplotlib import pyplot as plt

svgp_version = "botorch"
# svgp_version = "pytorch"

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x_mean = torch.linspace(0, 1, 20)
# We'll assume the variance shrinks the closer we get to 1
train_x_stdv = torch.linspace(0.03, 0.01, 20)

# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x_mean * (2 * math.pi)) + torch.randn(train_x_mean.size()) * 0.2
inducing_points = torch.randn(10, 1)


f, ax = plt.subplots(1, 1, figsize=(8, 3))
ax.errorbar(train_x_mean, train_y, xerr=(train_x_stdv * 2), fmt="k*", label="Train Data")
ax.legend()

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

# this is for running the notebook in our testing framework
import os
smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 400

print("Using ", svgp_version)

class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# pytorch version
if svgp_version != "botorch":
    model = GPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

# Botorch wrapper
else:
    variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
    # variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)

    model = SingleTaskVariationalGP(
                train_X=train_x_mean,
                num_outputs=1,
                variational_distribution=variational_distribution,
                inducing_points = inducing_points,
                learn_inducing_points=True,
                mean_module=gpytorch.means.ConstantMean(),
                covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()))

    likelihood = gpytorch.likelihoods.GaussianLikelihood()

model.train()
likelihood.train()

optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.01)

# Our loss object. We're using the VariationalELBO
if svgp_version != "botorch":
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))
else:
    mll = VariationalELBO(likelihood, model.model, num_data=10)

# iterator = tqdm.notebook.tqdm(range(training_iter))
for i in range(0, training_iter):
    # First thing: draw a sample set of features from our distribution
    train_x_sample = torch.distributions.Normal(train_x_mean, train_x_stdv).rsample()
    
    # Now do the rest of the training loop
    optimizer.zero_grad()
    output = model(train_x_sample)
    loss = -mll(output, train_y)
    # iterator.set_postfix(loss=loss.item())
    loss.backward()
    optimizer.step()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    observed_pred = likelihood(model(test_x))

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(8, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.errorbar(train_x_mean.numpy(), train_y.numpy(), xerr=train_x_stdv, fmt='k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

    plt.show()
