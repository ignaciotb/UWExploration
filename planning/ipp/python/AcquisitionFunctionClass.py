from botorch.acquisition import UpperConfidenceBound, MCAcquisitionFunction, AnalyticAcquisitionFunction
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor, cat
import torch
import dubins
from typing import Dict, Optional, Tuple, Union
import open3d as o3d
import numpy as np
import math
from typing import Any, Optional, Union
from botorch.acquisition.objective import (
    MCAcquisitionObjective,
    PosteriorTransform,
)
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.utils.transforms import (
    concatenate_pending_points,
    t_batch_mode_transform,
)
from torch import Tensor

class qUCB_xy(MCAcquisitionFunction):
    r"""MC-based batch Upper Confidence Bound.

    Uses a reparameterization to extend UCB to qUCB for q > 1 (See Appendix A
    of [Wilson2017reparam].)

    `qUCB = E(max(mu + |Y_tilde - mu|))`, where `Y_tilde ~ N(mu, beta pi/2 Sigma)`
    and `f(X)` has distribution `N(mu, Sigma)`.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> sampler = SobolQMCNormalSampler(1024)
        >>> qUCB = qUpperConfidenceBound(model, 0.1, sampler)
        >>> qucb = qUCB(test_X)
    """

    def __init__(
        self,
        model: Model,
        beta: float,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCAcquisitionObjective] = None,
        posterior_transform: Optional[PosteriorTransform] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        r"""q-Upper Confidence Bound.

        Args:
            model: A fitted model.
            beta: Controls tradeoff between mean and standard deviation in UCB.
            sampler: The sampler used to draw base samples. See `MCAcquisitionFunction`
                more details.
            objective: The MCAcquisitionObjective under which the samples are
                evaluated. Defaults to `IdentityMCObjective()`.
            posterior_transform: A PosteriorTransform (optional).
            X_pending: A `batch_shape x m x d`-dim Tensor of `m` design points that have
                points that have been submitted for function evaluation but have not yet
                been evaluated. Concatenated into X upon forward call. Copied and set to
                have no gradient.
        """
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            posterior_transform=posterior_transform,
            X_pending=X_pending,
        )
        self.beta_prime = math.sqrt(beta * math.pi / 2)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate qUpperConfidenceBound on the candidate set `X`.

        Args:
            X: A `batch_sahpe x q x d`-dim Tensor of t-batches with `q` `d`-dim design
                points each.

        Returns:
            A `batch_shape'`-dim Tensor of Upper Confidence Bound values at the given
            design points `X`, where `batch_shape'` is the broadcasted batch shape of
            model and input `X`.
        """
        posterior = self.model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(samples, X=X)
        mean = obj.mean(dim=0)
        ucb_samples = mean + self.beta_prime * (obj - mean).abs()
        return ucb_samples.max(dim=-1)[0].mean(dim=0)
    

class UCB_xy(UpperConfidenceBound):
    def __init__(
        self,
        model: Model,
        beta: Union[float, Tensor],
        posterior_transform: Optional[PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super(UCB_xy, self).__init__(model, beta, posterior_transform = None, maximize = False, **kwargs)
        self.register_buffer("beta", torch.as_tensor(beta))
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        mean, sigma = self._mean_and_sigma(X)
        return (abs(mean - self.model.model.mean_module.constant)) + self.beta * sigma


class UCB_path(AnalyticAcquisitionFunction):
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
            
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
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
    
    def _dubins_swath(self, xy, theta) -> Tuple[Tensor, Optional[Tensor]]:
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
        rewards = Tensor()
        
        for idx, place in enumerate(destinations):
            # Calculate dubins path to candidate, and travel cost
            path = dubins.shortest_path(self.current_state, [place[0], place[1], angles[idx]], self.turning_radius)
            wp_poses, length_arr = path.sample_many(self.wp_resolution)
            cost = length_arr[-1] + self.wp_resolution
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
            sigma = sigma.sum()
            #ucb = abs(mean - self.model.model.mean_module.constant) + self.beta.sqrt() * sigma #relative to gp mean
            ucb = sigma
            reward = torch.div(ucb, cost)
            rewards = cat((rewards,reward.reshape(1)),0)
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
        samples = Tensor()
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
                samples = cat((samples,Tensor([n1, n2])),0)

        return samples
    
    