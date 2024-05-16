# Torch libraries
import torch
import botorch
import gpytorch
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)

# Math libraries
import open3d as o3d
import numpy as np  
import dubins

# Python functionality      
import typing   

class BayesianOptimizer():
    """ Defines methods for BO optimization
    """
    def __init__(self, current_pose, gp_terrain, wp_resolution, turning_radius,
                 swath_width, path_nbr_samples, voxel_size, wp_sample_interval):
        """ Constructor method

        Args:
            gp_terrain (SingleTaskVariationalGP): SVGP that learns environment model from MBES input.
            current_pose (list[double]):          2D pose from where to start searches for new paths
            wp_resolution (double):               Resolution for calculating path waypoints
            turning_radius (double):              Turning radius of vehicle
            swath_width (float):                  Width of MBES sensor swath
            path_nbr_samples (int):               Number of orthogonal samples that emulate an MBES swath
            voxel_size (double):                  Size of grids that to reduce number of overlapping samples
            wp_sample_interval (int):             Interval of sampling waypoint swaths orthogonally along path
        """

        # Set up bounds, and environment acq_fun
        self.current_pose           = current_pose
        self.bounds_theta_torch     = torch.tensor([[-np.pi],  [np.pi]]).to(torch.float)
        self.path_reward            = UCB_path(model=gp_terrain, current_pose=current_pose, wp_resolution=wp_resolution,
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
        self.gp_theta               = botorch.models.SingleTaskGP(train_X, train_Y)
        self.mll                    = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_theta.likelihood, self.gp_theta)
        botorch.fit.fit_gpytorch_mll(self.mll)
        best_theta_acqf             = botorch.acquisition.PosteriorMean(self.gp_theta)
        _, value                    = botorch.optim.optimize_acqf(best_theta_acqf, bounds=self.bounds_theta_torch, q=1, num_restarts=10, raw_samples=20)
        self.theta_acqf             = botorch.acquisition.ExpectedImprovement(self.gp_theta, value)
        
        
        # Run BO to iteratively sample new paths, to improve our chances of optimal heading
        iteration = 0
        while iteration < max_iter:
            candidate, value = botorch.optim.optimize_acqf(self.theta_acqf, bounds=self.bounds_theta_torch, q=1, num_restarts=5, raw_samples=20)
            
            # TODO: For optimizng radius - pass radius to get training data
            sample = torch.cat([XY, candidate], 1).squeeze(0)
            train_X, train_Y  = self._sample_paths(nbr_samples=1, X=sample)
            train_X = train_X[2].unsqueeze(0)
            
            self.gp_theta = self.gp_theta.get_fantasy_model(train_X, train_Y)

            self.theta_acqf.model = self.gp_theta
            iteration += 1
        
        
        # Run a single optimization with no regard for variance, only caring for highest mean
        
        # TODO: For optimizng radius - pass radius to get training data
        best_theta_acqf  = botorch.acquisition.PosteriorMean(self.gp_theta)
        best_candidate, value = botorch.optim.optimize_acqf(self.best_theta_acqf, bounds=self.bounds_theta_torch, q=1, num_restarts=20, raw_samples=20)
                
        # TODO: For optimizng radius - ensure we return both a best theta, and radius
        return best_candidate, self.gp_theta
    

class UCB_path(botorch.acquisition.AnalyticAcquisitionFunction):
    def __init__(self, model, beta, current_pose, wp_resolution, turning_radius, swath_width, path_nbr_samples, 
                 voxel_size = 3, wp_sample_interval = 6, posterior_transform = None, **kwargs):
        
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        
        self.current_state = current_pose
        self.wp_resolution = wp_resolution
        self.wp_sample_interval = int(wp_sample_interval)
        self.turning_radius = turning_radius
        self.swath_width = swath_width
        self.nbr_samples = path_nbr_samples
        self.voxel_size = voxel_size
        self.beta = beta
            
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X. (note, limited to
        q = 1, single candidate output)

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        # Split suggested planar coordinates and headings
        xy = X[:,:,:2]
        theta = X[:,:,2]
        

        # Get the reward along path associated with travelling to candidates
        rewards = self._dubins_swath(xy, theta)

        return rewards
    
    def _dubins_swath(self, xy, theta) -> typing.Tuple[torch.Tensor, typing.Optional[torch.Tensor]]:
        """ Computes the dubins path to the candidates. Generates points
            along the line to be used for posterior sampling, and calculates
            the cost of the path as the length.

        Args:
            xy: `batch_shape x q x (d-1)`-dim Tensor of model inputs.
            theta: `batch_shape x q x 1`-dim Tensor of model inputs.

        Returns:
            A tuple of tensors containing the cost of travelling. 
            Removes the last two dimensions if they have size one.
        """
        destinations = (xy.squeeze(-2).squeeze(-1))
        angles = (theta.squeeze(-1))
        rewards = torch.Tensor()
        
        for idx, place in enumerate(destinations):
            # Calculate dubins path to candidate, and travel cost
            path = dubins.shortest_path(self.current_state, [place[0], place[1], angles[idx]], self.turning_radius)
            wp_poses, length_arr = path.sample_many(2)
            cost = (length_arr[-1] + 2) ** 1.5
            # Get sample swath points orthogonally to path at regular intervals
            points = self._get_orthogonal_samples(wp_poses[::self.wp_sample_interval], self.nbr_samples, self.swath_width)
            # Voxelize in 2D to get even spread
            pcl = np.array(points)
            b = np.zeros((pcl.shape[0], pcl.shape[1] + 1))
            b[:,:-1] = pcl
            pcd3 = o3d.geometry.PointCloud()
            pcd3.points = o3d.utility.Vector3dVector(b)

            pcd3 = pcd3.voxel_down_sample(self.voxel_size)
            #o3d.visualization.draw_geometries([pcd3])
            xyz = np.asarray(pcd3.points)

            xy = torch.from_numpy(xyz[:, :2]).type(torch.FloatTensor)
            
            # Calculate UCB/cost reward of travelling to candidate
            _, sigma = self._mean_and_sigma(xy)
            #mean = mean.sum()
            sigma = sigma.sum() * 1000
            #ucb = abs(mean - self.model.model.mean_module.constant) + self.beta.sqrt() * sigma #relative to gp mean
            ucb = sigma
            reward = torch.div(ucb, cost)
            rewards = torch.cat((rewards,reward.reshape(1)),0)
        return rewards
    
    def _get_orthogonal_samples(self, poses, nbr_samples=6, swath_width=5.0):
        """ Generates points on lines orthogonal to a vector. Will generate
            `nbr_samples` for each given vector, along a line of given swath width.
        

        Args:
            poses (list[float]): [x y theta]
            nbr_samples (int, optional): number of samples generated for each vector. Defaults to 6.
            swath_width (float, optional): width of line sampled from, for each vector. Defaults to 5.0.

        Returns:
            torch.Tensor: concatenated xy points of samples
        """
        samples = torch.Tensor()
        radius = 0.5*swath_width
        for pose in poses:
            x = pose[0]
            y = pose[1]
            yaw = pose[2]

            dx = radius*np.sin(yaw) # shifted by 90 degree for orthogonality
            dy = radius*np.cos(yaw) 

            for i in np.linspace(-1, 1, nbr_samples):
                dx_s = dx * i
                dy_s = dy * i
                n1 = [x + dx_s, y - dy_s]
                n2 = [x - dx_s, y + dy_s]
                #all_samples.append(np.array([n1, n2]))
                samples = torch.cat((samples,torch.Tensor([n1, n2])),0)

        return samples