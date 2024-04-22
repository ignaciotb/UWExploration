# Torch libraries
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound, qUpperConfidenceBound
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf

# Custom module
from AcquisitionFunctionClass import UCB_custom

# Numpy and python imports
import numpy as np
import cma                      
import time

class BayesianOptimizer():
    """ Defines methods for BO optimization
    """
    def __init__(self, nbr_initial_samples, gp_terrain, bounds, beta, current_pose):
        """ Constructor method

        Args:
            nbr_initial_samples (int):                  Initial samples to populate second GP with
            gp_terrain (gpytorch.models.VariationalGP): Gaussian process model for environment
            bounds (array[double]):                     Boundaries of suggested candidates
            beta (double):                              UCB parameter
            current_pose (list[float]):                 2D pose of current position of vehicle
        """
        # Set up bounds, and environment acq_fun
        self.beta                   = beta
        self.bounds_torch           = torch.tensor([[bounds[0], bounds[3], -np.pi], [bounds[1], bounds[2], np.pi]]).to(torch.float)
        self.bounds_list            = [[bounds[0], bounds[3], -np.pi], [bounds[1], bounds[2], np.pi]]
        self.path_reward            = UCB_custom(gp_terrain, self.beta, current_pose)
        
        # Setup second layer of GP, train it
        self.train_X, self.train_Y  = self._sample_paths(nbr_samples=nbr_initial_samples)
        self.gp_path                = SingleTaskGP(self.train_X, self.train_Y, outcome_transform=Standardize(m=1))
        self.mll                    = ExactMarginalLogLikelihood(self.gp_path.likelihood, self.gp_path)
        fit_gpytorch_mll(self.mll)
        self.gp_path_acqf           = UpperConfidenceBound(self.gp_path, self.beta)
    
    def _sample_paths(self, nbr_samples, X=None):
        """ Sample the reward of swaths along dubins path with environment GP.
            Primarily used to generate data train the second layer of GP. 
            If not given data, generates random data inside bounds and calculates
            reward. If given data, will calculate the target reward for that data.

        Args:
            nbr_samples (int): number of train and target data points to sample

        Returns:
            [nbr_samples x 3], [nbr_samples x 1]: tensors containing train and target data
        """
        
        # Disable gradients, otherwise fit_gpytorch_mll throws a fit later
        with torch.no_grad():
            if X is None:
                train_X = (torch.from_numpy(np.random.uniform(low=self.bounds_list[0], 
                                    high=self.bounds_list[1], size=[nbr_samples, 3]))).type(torch.FloatTensor)
            else:
                train_X = X
            train_Y = (self.path_reward.forward(train_X.unsqueeze(-2))).unsqueeze(1)
        return train_X, train_Y        
        
    
    def optimize_with_grad(self, max_iter=5):
        """ Returns the most optimal candidate to move towards in a dubins path,
        to gather information. Uses first order (gradient) based optimization, which is
        enabled by having a second layer of GP that learns the rewards of the paths.

        Args:
            max_iter (int, optional): Number of iterations of BO to use as budget. Defaults to 10.

        Returns:
            (list[double, double, double]): Candidate pose in 2D [x, y, yaw]
            botorch.models.SingleTaskGP: The trained second layer GP, if needed for debugging
        """
        iteration = 0
        best_value = 0
        
        # Each iteration, optimize acq fun. of path GP to generate a candidate (x,y,yaw).
        # Then evaluate path connecting to that candidate, in environment GP. Add this data,
        # and train the path GP on this, then loop. Return best candidate.
        
        # TODO: Training this exact GP is slow. Fixes: train batch (q candidates) or other (fantasy) model? 
        while iteration < max_iter:
            candidate, value = optimize_acqf(self.gp_path_acqf, bounds=self.bounds_torch, q=1, num_restarts=3, raw_samples=50)
            train_X, train_Y  = self._sample_paths(nbr_samples=1, X=candidate)
            
            # TODO: need to check if fantasy model works as expected
            self.gp_path = self.gp_path.get_fantasy_model(train_X, train_Y)
            #self.train_X = torch.cat((self.train_X,train_X),0)
            self.gp_path_acqf.model = self.gp_path
            if value > best_value:
                print("changed best value!")
                best_value = value
                best_candidate = candidate
            iteration += 1
            
        return best_candidate, self.gp_path
    
    def optimize_no_grad(self):
        """ Zeroth-order optimization, using CMA-ES. This directly optimizes on the environment
            GP, by using the full path acquisition function.

        Returns:
            (list[float, float, float], float): tuple with candidate 2D pose, and value of candidate
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
