#!/usr/bin/env python3

import os
import torch
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from gpytorch.models import VariationalGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel, ScaleKernel, GaussianSymmetrizedKLKernel, InducingPointKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood, ExactMarginalLogLikelihood
from gpytorch.test.utils import least_used_cuda_device
import gpytorch.settings
#from convergence import ExpMAStoppingCriterion
import matplotlib.pyplot as plt


class SVGP(VariationalGP):

    def __init__(self, num_inducing):

        # variational distribution and strategy
        # NOTE: we put random normal dumby inducing points
        # here, which we'll change in self.fit
        vardist = CholeskyVariationalDistribution(num_inducing)
        varstra = VariationalStrategy(
            self,
            torch.randn((num_inducing, 2)),
            vardist,
            learn_inducing_locations=True
        )
        VariationalGP.__init__(self, varstra)

        # kernel — implemented in self.forward
        self.mean = ConstantMean()
        self.cov = MaternKernel(ard_num_dims=2)
        # self.cov = GaussianSymmetrizedKLKernel()
        self.cov = ScaleKernel(self.cov, ard_num_dims=2)

    def forward(self, input):
        m = self.mean(input)
        v = self.cov(input)
        return MultivariateNormal(m, v)


def plot_post(cp, inputs, targets, fname, n=80, n_contours=50):

    # Reconstruct model
    model = SVGP(400)
    likelihood = GaussianLikelihood()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    likelihood.to(device).float()
    model.to(device).float()

    model.load_state_dict(cp['model'])
    likelihood.load_state_dict(cp['likelihood'])

    # sanity
    assert inputs.shape[0] == targets.shape[0]
    assert inputs.shape[1] == 2

    # toggle evaluation mode
    likelihood.eval()
    model.eval()
    torch.cuda.empty_cache()

    # posterior sampling locations
    inputsg = [
        np.linspace(min(inputs[:, 0]), max(inputs[:, 0]), n),
        np.linspace(min(inputs[:, 1]), max(inputs[:, 1]), n)
    ]
    inputst = np.meshgrid(*inputsg)
    s = inputst[0].shape
    inputst = [_.flatten() for _ in inputst]
    inputst = np.vstack(inputst).transpose()
    inputst = torch.from_numpy(inputst).to(device).float()

    # sample
    with torch.no_grad():
        outputs = model(inputst)
        outputs = likelihood(outputs)
        mean = outputs.mean.cpu().numpy().reshape(s)
        variance = outputs.variance.cpu().numpy().reshape(s)

    # plot raw, mean, and variance
    levels = np.linspace(min(targets)-20, max(targets)+20, n_contours)
    fig, ax = plt.subplots(3, sharex=True, sharey=True)
    cr = ax[0].scatter(inputs[:, 0], inputs[:, 1], c=targets,
                        cmap='viridis', s=0.4, edgecolors='none')
    cm = ax[1].contourf(*inputsg, mean, levels=n_contours)
    cv = ax[2].contourf(*inputsg, variance, levels=n_contours)
    indpts = model.variational_strategy.inducing_points.data.cpu().numpy()
    ax[2].plot(indpts[:, 0], indpts[:, 1], 'ko', markersize=1, alpha=0.2)

    # colorbars
    fig.colorbar(cr, ax=ax[0])
    fig.colorbar(cm, ax=ax[1])
    fig.colorbar(cv, ax=ax[2])

    # formatting
    ax[0].set_aspect('equal')
    ax[0].set_title('Raw data')
    ax[0].set_ylabel('$y~[m]$')
    ax[1].set_aspect('equal')
    ax[1].set_title('Mean')
    ax[1].set_ylabel('$y~[m]$')
    ax[2].set_aspect('equal')
    ax[2].set_title('Variance')
    ax[2].set_xlabel('$x~[m]$')
    ax[2].set_ylabel('$y~[m]$')
    plt.tight_layout()

    # save
    fig.savefig(fname, bbox_inches='tight', dpi=1000)

    # plot_loss(self.storage_path + 'particle_' + str(self.particle_id)
    #                 + '_training_loss.png', loss)

    # Free up GPU mem
    del inputst
    torch.cuda.empty_cache()

    def plot_loss(fname, loss):

        # plot
        fig, ax = plt.subplots(1)
        ax.plot(loss, 'k-')

        # format
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ELBO')
        ax.set_yscale('log')
        plt.tight_layout()

        # save
        fig.savefig(fname, bbox_inches='tight', dpi=1000)


if __name__ == '__main__':

    i = str(input("Number of the particle to plot: "))

    cp = torch.load(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'rbpf/svpg_final_'+i+'.pth')))
    data = np.load(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'rbpf/map_'+i+'.npz')))

    beams = data['beams']
    # loss = data['loss']

    plot_post(cp, beams[:, 0:2], beams[:, 2], './particle_map_' + i + '.png', n=50, n_contours=100)
