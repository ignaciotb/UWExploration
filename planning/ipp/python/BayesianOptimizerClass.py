# Optimization and BO libraries
import torch
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from AcquisitionFunctionClass import UCB_custom
import numpy as np
import cma
import time

class BayesianOptimizer():
    """ Defines methods for BO optimization
    """
    def __init__(self, gp, bounds, beta, current_pose):
        """ Constructor method

        Args:
            gp  (gpytorch.models.VariationalGP): Gaussian process model
            bounds              (array[double]): Boundaries of suggested candidates
        """
        # TODO: bounds[:][2] should be half a pi up and down from current yaw 
        self.bounds_torch = torch.tensor([[bounds[0], bounds[3], -np.pi], [bounds[1], bounds[2], np.pi]]).to(torch.float)
        self.bounds_list  = [[bounds[0], bounds[3], -np.pi], [bounds[1], bounds[2], np.pi]]
        self.aq_func      = UCB_custom(gp, beta, current_pose)
    
    def optimize_no_grad(self):
        """ Zeroth-order optimization, using CMA-ES. 

        Returns:
            _type_: _description_
        """
        x0 = np.random.uniform(low=self.bounds_list[0], 
                               high=self.bounds_list[1], size=3)
        es = cma.CMAEvolutionStrategy(x0=x0, sigma0=20, inopts={"bounds": self.bounds_list, "popsize": 200, "maxiter": 10})
        with torch.no_grad():

            # Run the optimization loop using the ask/tell interface -- this uses
            # PyCMA's default settings, see the PyCMA documentation for how to modify these
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            while not es.stop():
                xs = es.ask()  # as for new points to evaluate
                # convert to Tensor for evaluating the acquisition function
                X = torch.tensor(xs, device=device, dtype=torch.float)
                # evaluate the acquisition function (optimizer assumes we're minimizing)
                Y = -self.aq_func(X.unsqueeze(-2))  # acquisition functions require an explicit q-batch dimension
                y = Y.view(-1).double().numpy()  # convert result to numpy array
                es.tell(xs, y)  # return the result to the optimizer
                print("iteration")

        # convert result back to a torch tensor
        best_x = torch.from_numpy(es.best.x).to(X)
        best_y = es.best.f

        return best_x, best_y
    
    def optimize__with_grad(self):
        """ Returns the most optimal candidate for sampling

        Returns:
            ([double, double], double): Candidate location and expected value of sampling that location
        """
        candidate, acq_value = optimize_acqf(self.aq_func, bounds=self.bounds, q=1, num_restarts=5, raw_samples=10)
        return candidate, acq_value


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
