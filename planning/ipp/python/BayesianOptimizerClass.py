# Torch libraries
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP
from botorch.acquisition import UpperConfidenceBound, PosteriorMean
from botorch.optim import optimize_acqf

# Custom module
from AcquisitionFunctionClass import UCB_path, UCB_xy

# Numpy and python imports
import numpy as np                   

class BayesianOptimizer():
    """ Defines methods for BO optimization
    """
    def __init__(self, gp_terrain, bounds, beta, current_pose, wp_resolution, turning_radius,
                 swath_width, path_nbr_samples, voxel_size, wp_sample_interval):
        """ Constructor method

        Args:
            gp_terrain (SingleTaskVariationalGP): SVGP that learns environment model from MBES input.
            bounds (array[double]):               Boundaries of where vehicle should travel
            beta (double):                        Constant for exploration vs exploitation, DEPRECATED
            current_pose (list[double]):          2D pose from where to start searches for new paths
            wp_resolution (double):               Resolution for calculating path waypoints
            turning_radius (double):              Turning radius of vehicle
            swath_width (float):                  Width of MBES sensor swath
            path_nbr_samples (int):               Number of orthogonal samples that emulate an MBES swath
            voxel_size (double):                  Size of grids that to reduce number of overlapping samples
            wp_sample_interval (int):             Interval of sampling waypoint swaths orthogonally along path
        """

        # Set up bounds, and environment acq_fun
        self.beta                   = beta
        self.current_pose           = current_pose
        self.bounds_XY_torch        = torch.tensor([[bounds[0], bounds[3]], [bounds[1], bounds[2]]]).to(torch.float)
        self.bounds_theta_torch     = torch.tensor([[-np.pi],  [np.pi]]).to(torch.float)
        self.XY_acqf                = UCB_xy(model=gp_terrain, beta=self.beta)
        self.path_reward            = UCB_path(model=gp_terrain, beta=self.beta, current_pose=current_pose, wp_resolution=wp_resolution,
                                               turning_radius=turning_radius, swath_width=swath_width, path_nbr_samples=path_nbr_samples,
                                               voxel_size=voxel_size, wp_sample_interval=wp_sample_interval)

    
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
        
    def optimize_XY_with_grad(self, max_iter=5, nbr_samples=200):
        """ Finds the optimal sampling location in 2D from the environment GP

        Args:
            max_iter (int, optional): max number of restarts for gradient optimization. Defaults to 5.
            nbr_samples (int, optional): raw samples used in gradient optimization. Defaults to 200.

        Returns:
            Tensor: best candidate(s) found from optimization
        """
        candidates, _ = optimize_acqf(acq_function=self.XY_acqf, bounds=self.bounds_XY_torch, q=1, num_restarts=max_iter, raw_samples=nbr_samples)
        return candidates
    
    def optimize_theta_with_grad(self, XY, max_iter=5, nbr_samples=10):
        """ Finds the optimal heading to connect current position with the
            best sampling location, using a 1D GP that learns rewards by
            sampling paths (representing different headings) in the environment GP.

        Args:
            XY (Tensor): `[X, Y]` 2D position of sampling location to connect to
            max_iter (int, optional): number of loops in BO where new paths are sampled. Defaults to 5.
            nbr_samples (int, optional): number of initial sampled paths from environment GP. Defaults to 10.

        Returns:
            Tensor: best candidate found from optimization
        """

        # Generate initial sample angles
                
        # TODO: For optimizng radius - generate theta and radius
        random_thetas = torch.linspace(self.bounds_theta_torch[0,0].item(), self.bounds_theta_torch[1,0].item(), nbr_samples).unsqueeze(1)
        XY_repeated = XY.repeat(nbr_samples, 1)
        samples = torch.cat([XY_repeated, random_thetas], 1)
        
        
        # TODO: For optimizng radius - pass radius to get training data
        train_X, train_Y  = self._sample_paths(nbr_samples=nbr_samples, X=samples)
        train_X = train_X[:, 2].unsqueeze(1)
        
        
        # Train a new 1D Gaussian process for rewards of different headings
        self.gp_theta               = SingleTaskGP(train_X, train_Y)
        self.mll                    = ExactMarginalLogLikelihood(self.gp_theta.likelihood, self.gp_theta)
        fit_gpytorch_mll(self.mll)
        self.theta_acqf             = UpperConfidenceBound(self.gp_theta, self.beta)
        
        
        # Run BO to iteratively sample new paths, to improve our chances of optimal heading
        iteration = 0
        while iteration < max_iter:
            candidate, value = optimize_acqf(self.theta_acqf, bounds=self.bounds_theta_torch, q=1, num_restarts=5, raw_samples=20)
            
            # TODO: For optimizng radius - pass radius to get training data
            sample = torch.cat([XY, candidate], 1).squeeze(0)
            train_X, train_Y  = self._sample_paths(nbr_samples=1, X=sample)
            train_X = train_X[2].unsqueeze(0)
            
            self.gp_theta = self.gp_theta.get_fantasy_model(train_X, train_Y)

            self.theta_acqf.model = self.gp_theta
            iteration += 1
        
        
        # Run a single optimization with no regard for variance, only caring for highest mean
        
        # TODO: For optimizng radius - pass radius to get training data
        self.best_theta_acqf  = PosteriorMean(self.gp_theta)
        best_candidate, value = optimize_acqf(self.best_theta_acqf, bounds=self.bounds_theta_torch, q=1, num_restarts=20, raw_samples=20)
                
        # TODO: For optimizng radius - ensure we return both a best theta, and radius
        return best_candidate, self.gp_theta