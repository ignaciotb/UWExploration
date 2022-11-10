#!/usr/bin/env python3

# Script for finding good stopping conditions for the training

import numpy as np
import torch
from gp_mapping.convergence import ExpMAStoppingCriterion
import matplotlib.pyplot as plt
import sys

def plot_loss(fname, loss_1, loss_2):

    # plot
    fig, ax = plt.subplots(1)
    ax.plot(loss_1, 'r-')
    ax.plot(loss_2, 'k-')

    # format
    ax.set_xlabel('Iteration')
    ax.set_ylabel('ELBO')
    ax.set_yscale('log')
    plt.tight_layout()

    # save
    fig.savefig(fname, bbox_inches='tight', dpi=1000)


data = np.load(sys.argv[1]) 
loss_1 = data["loss"]
# loss_2 = np.load(sys.argv[2]) 


loss_tensor_1 = torch.from_numpy(loss_1)
# loss_tensor_2 = torch.from_numpy(loss_2)

#plot_loss("./test_loss.png", loss_1, loss_2)

# criterion = ExpMAStoppingCriterion(rel_tol=1e-12, minimize=True, n_window=2000) 
# print("UUI")
# for i in range(0,len(loss_1)): 
#     if criterion.evaluate(loss_tensor_1[i]): 
#         print(i) 
#         break

criterion = ExpMAStoppingCriterion(rel_tol=1e-1, minimize=True, n_window=50) 
print("Total iterations ", len(loss_tensor_1))
for i in range(0,len(loss_1)): 
    if criterion.evaluate(loss_tensor_1[i]): 
        print("Stopping here ", i)
        break

