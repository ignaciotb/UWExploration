# Optimization and BO libraries
import torch
import botorch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from AcquisitionFunctionClass import UCB_custom
import numpy as np
import cma
import time

class BayesianOptimizer():
    """ Defines methods for BO optimization
    """
    def __init__(self, gp_terrain, bounds, beta, current_pose):
        """ Constructor method

        Args:
            gp  (gpytorch.models.VariationalGP): Gaussian process model
            bounds              (array[double]): Boundaries of suggested candidates
        """
        self.bounds_torch = torch.tensor([[bounds[0], bounds[3], -np.pi], [bounds[1], bounds[2], np.pi]]).to(torch.float)
        self.bounds_list  = [[bounds[0], bounds[3], -np.pi], [bounds[1], bounds[2], np.pi]]
        self.path_reward  = UCB_custom(gp_terrain, beta, current_pose)
        
        
        self.train_X, self.train_Y  = self._sample_paths(nbr_samples=100)
        self.gp_path      = SingleTaskGP(self.train_X, self.train_Y, outcome_transform=Standardize(m=1))
        self.mll = ExactMarginalLogLikelihood(self.gp_path.likelihood, self.gp_path)
        fit_gpytorch_mll(self.mll)
        self.gp_path_acqf = UpperConfidenceBound(self.gp_path, 10)
    
    def _sample_paths(self, nbr_samples, X=None):
        """_summary_

        Args:
            nbr_samples (_type_): _description_

        Returns:
            _type_: _description_
        """
        with torch.no_grad():
            if X is None:
                train_X = (torch.from_numpy(np.random.uniform(low=self.bounds_list[0], 
                                    high=self.bounds_list[1], size=[nbr_samples, 3]))).type(torch.FloatTensor)
            else:
                train_X = X
            train_Y = (self.path_reward.forward(train_X.unsqueeze(-2))).unsqueeze(1)
        return train_X, train_Y        
        
    
    def optimize_with_grad(self, max_iter=10):
        """ Returns the most optimal candidate to move towards

        Returns:
            ([double, double, double]): Candidate pose in 2D [x, y, yaw]
        """
        iteration = 0
        best_value = 0
        while iteration < max_iter:
            #t1 = time.time()
            candidate, value = optimize_acqf(self.gp_path_acqf, bounds=self.bounds_torch, q=1, num_restarts=5, raw_samples=40)
            train_X, train_Y  = self._sample_paths(nbr_samples=1, X=candidate)
            #t2 = time.time()
            self.train_X = torch.cat((self.train_X,train_X),0)
            self.train_Y = torch.cat((self.train_Y,train_Y),0)
            self.gp_path = SingleTaskGP(self.train_X, self.train_Y, outcome_transform=Standardize(m=1))
            fit_gpytorch_mll(ExactMarginalLogLikelihood(self.gp_path.likelihood, self.gp_path))
            #t3 = time.time()
            if value > best_value:
                best_value = value
                best_candidate = candidate
            iteration += 1
            #print(t2-t1)
            #print(t3-t2)
            
        return best_candidate
    
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
