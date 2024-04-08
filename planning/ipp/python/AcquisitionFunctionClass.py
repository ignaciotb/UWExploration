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
        mean, sigma = self._mean_and_sigma(X)
        ucb = (mean if self.maximize else -mean) + self.beta.sqrt() * sigma
        cost = self._travel_cost(X, self.current_state)
        reward = torch.div(ucb, cost)
        return ucb
    
    def _travel_cost(self, X: Tensor, current_pose) -> Tuple[Tensor, Optional[Tensor]]:
        """Computes the first and second moments of the model posterior.

        Args:
            X: `batch_shape x q x d`-dim Tensor of model inputs.
            current_pose (_type_): _description_

        Returns:
            A tuple of tensors containing the cost of travelling. 
            Removes the last two dimensions if they have size one.
        """
        destinations = (X.squeeze(-2).squeeze(-1)).detach().numpy()
        #print(destinations)
        wp_resolution = 0.5
        turning_radius = 5
        costs = Tensor()
        for place in destinations:
            possible_paths = []
            for angle in np.linspace(0, 6.28, 4):
                path = dubins.shortest_path(current_pose, [place[0], place[1], angle], turning_radius)
                _, length_arr = path.sample_many(wp_resolution) # _ = waypoints, if needed for future reference
                cost = length_arr[-1] + wp_resolution
                possible_paths.append(cost)
            lowest_cost = min(possible_paths)
            costs = cat((costs,Tensor([lowest_cost])),0)
        return costs
    
    