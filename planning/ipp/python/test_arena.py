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


from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf


corners = [-250, -150, 0, -60]

n_ip = 10
minibatch = 5
lr = 0.01


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