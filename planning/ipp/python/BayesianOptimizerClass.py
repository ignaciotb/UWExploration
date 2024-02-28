# Optimization and BO libraries
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf

class BayesianOptimizer():
    """ Defines methods for BO optimization
    """
    def __init__(self, gp, bounds):
        """ Constructor method

        Args:
            gp  (gpytorch.models.VariationalGP): Gaussian process model
            bounds              (array[double]): Boundaries of suggested candidates
        """
        
        self.gp = gp
        self.bounds = torch.tensor([[bounds[0], bounds[3]], [bounds[1], bounds[2]]]).to(torch.float)
        self.aq_func = UpperConfidenceBound(self.gp, beta=0.1)
        
    def optimize(self):
        """ Returns the most optimal candidate for sampling

        Returns:
            ([double, double], double): Candidate location and expected value of sampling that location
        """
        candidate, acq_value = optimize_acqf(self.aq_func, bounds=self.bounds, q=1, num_restarts=5, raw_samples=20)
        return candidate, acq_value

    def update_gp(self, gp):
        """ Update the posterior GP used for sampling (needed before optimization)

        Args:
            gp  (gpytorch.models.VariationalGP): Gaussian process model
        """
        self.gp = gp
        self.aq_func = UpperConfidenceBound(self.gp, beta=0.1)


## Old shit below, used during development for testing

"""
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood

train_X = torch.rand(10, 2)
Y = 1 - torch.linalg.norm(train_X - 0.5, dim=-1, keepdim=True)
Y = Y + 0.1 * torch.randn_like(Y)  # add some noise
train_Y = standardize(Y)

gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)

UCB = UpperConfidenceBound(gp, beta=0.1)

#bounds = torch.stack([torch.zeros(2), torch.ones(2)])
#candidate, acq_value = optimize_acqf(
#    UCB, bounds=bounds, q=1, num_restarts=5, raw_samples=20,
#)
#print(candidate)  # tensor([0.4887, 0.5063])
        
BO = BayesianOptimizer(gp=gp, bounds=[0, 10, 10, 0])
candidate, value = BO.optimize()

print(candidate)
bounds=[0, 10, 10, 0]
a = torch.stack([torch.zeros(2), torch.ones(2)])
b = torch.tensor([[bounds[0], bounds[3]], [bounds[1], bounds[2]]])
print(a)
print(a.type())
print(b)
print(b.type())
c = b.to(torch.float)
print(c.type())
#print(candidate)
"""
