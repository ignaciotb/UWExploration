from botorch.acquisition import UpperConfidenceBound
from botorch.utils.transforms import t_batch_mode_transform
from torch import Tensor, cat
import torch
import dubins
from typing import Dict, Optional, Tuple, Union
import numpy as np

class UCB_custom(UpperConfidenceBound):
    def __init__(self, model, beta, current_pose, posterior_transform = None, maximize = True, **kwargs):
        super(UCB_custom, self).__init__(model, beta, posterior_transform = None, maximize = False, **kwargs)
        self.current_state = current_pose
            
    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

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
        #print(X)
        #print(rewards)
        #mean, sigma = self._mean_and_sigma(xy)
        #ucb = (mean if self.maximize else -mean) + self.beta.sqrt() * sigma
        #print(ucb)
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
        #print(destinations)
        wp_resolution = 2
        turning_radius = 8
        rewards = Tensor()
        for idx, place in enumerate(destinations):
            # Calculate dubins path to candidate, and travel cost
            path = dubins.shortest_path(self.current_state, [place[0], place[1], angles[idx]], turning_radius)
            wp_poses, length_arr = path.sample_many(wp_resolution)
            cost = length_arr[-1] + wp_resolution
            
            # Get sample swath points orthogonally to path at regular intervals
            points = self._get_orthogonal_samples(wp_poses)
            
            # Calculate UCB/cost reward of travelling to candidate
            mean, sigma = self._mean_and_sigma(points)
            mean = mean.sum()
            sigma = sigma.sum()
            ucb = (mean if self.maximize else -mean) + self.beta.sqrt() * sigma
            reward = torch.div(ucb, cost)
            rewards = cat((rewards,reward.reshape(1)),0)
        return rewards
    
    def _get_orthogonal_samples(self, poses):
        samples = Tensor()
        for pose in poses:
            x = pose[0]
            y = pose[1]
            yaw = pose[2]

            dx = 4*np.sin(yaw) # shifted by 90 degree for orthogonality
            dy = 4*np.cos(yaw) 

            for i in np.linspace(0.2, 1, 3):
                dx_s = dx * i
                dy_s = dy * i
                n1 = [x + dx_s, y - dy_s]
                n2 = [x - dx_s, y + dy_s]
                #all_samples.append(np.array([n1, n2]))
                samples = cat((samples,Tensor([n1, n2])),0)

        return samples
    
    