#!/usr/bin/env python3

import torch
import numpy as np
import matplotlib.pyplot as plt
from gpytorch.models import VariationalGP, ExactGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel, ScaleKernel, GaussianSymmetrizedKLKernel, InducingPointKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls import VariationalELBO, PredictiveLogLikelihood, ExactMarginalLogLikelihood
import gpytorch.settings
#from convergence import ExpMAStoppingCriterion
from gp_mapping.convergence import ExpMAStoppingCriterion
import matplotlib.pyplot as plt

# This is not tested
class RGP(ExactGP):

    def __init__(self, inputs, targets, likelihood):

        # check the hardware
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # store inputs and outputs
        self.inputs = torch.from_numpy(inputs).float().to(self.device)
        self.targets = torch.from_numpy(targets).float().to(self.device)

        # initialise GP and store likelihood
        ExactGP.__init__(self, self.inputs, self.targets, likelihood)
        self.likelihood = likelihood

        # mean and covariance
        self.mean = ConstantMean()
        # self.cov = GaussianSymmetrizedKLKernel()
        self.cov = MaternKernel(ard_num_dims=2)
        self.cov = ScaleKernel(self.cov, ard_num_dims=2)
        self.cov = InducingPointKernel(self.cov, self.inputs, self.likelihood)

        # you better have a GPU!
        self.likelihood.to(self.device).float()
        self.to(self.device).float()

    def forward(self, inputs):
        mean = self.mean(inputs)
        cov = self.cov(inputs)
        return MultivariateNormal(mean, cov)

    def fit(self, max_iter=100, learning_rate=1e-3, rtol=1e-2, n_window=100, auto=False, verbose=True):

        # loss function
        mll = ExactMarginalLogLikelihood(self.likelihood, self)

        # stochastic optimiser
        opt = torch.optim.Adam(self.parameters(),lr=learning_rate)

        # convergence criterion
        if auto: criterion = ExpMAStoppingCriterion(rel_tol=rtol, n_window=n_window)

        # episode iteratior
        epochs = range(max_iter)
        epochs = tqdm.tqdm(epochs) if verbose else epochs

        # train
        self.train()
        self.likelihood.train()
        for _ in epochs:

            # compute loss, compute gradient, and update
            opt.zero_grad()
            loss = -mll(self(self.inputs), self.targets)
            loss.backward()
            opt.step()

            # verbosity and convergence check
            if verbose:
                epochs.set_description('Loss {:.4f}'.format(loss.item()))
            if auto and criterion.evaluate(loss.detach()):
                break

    def plot(self, fname, n=100, n_contours=50):

        '''
        Plots:
            ax[0]: raw inputs and targets,
            ax[1]: posterior predictive mean,
            ax[2]: posterior predictive variance
        inputs: (n,2) numpy array of inputs
        output: (n,) numpy array of targets
        fname: path to save plot at (extension determines file type, e.g. .png or .pdf)
        n: determines n² number of sampling locations to plot GP posterior
        n_contours: number of contours to show output magnitude with
        '''

        # toggle evaluation mode
        self.likelihood.eval()
        self.eval()
        torch.cuda.empty_cache()

        # posterior sampling locations
        inputs = self.inputs.cpu().numpy()
        targets = self.targets.cpu().numpy()
        inputsg = [
            np.linspace(min(inputs[:,0]), max(inputs[:,0]), n),
            np.linspace(min(inputs[:,1]), max(inputs[:,1]), n)
        ]
        inputst = np.meshgrid(*inputsg)
        s = inputst[0].shape
        inputst = [_.flatten() for _ in inputst]
        inputst = np.vstack(inputst).transpose()
        inputst = torch.from_numpy(inputst).to(self.device).float()

        # sample
        with torch.no_grad():
            outputs = self(inputst)
            outputs = self.likelihood(outputs)
            mean = outputs.mean.cpu().numpy().reshape(s)
            variance = outputs.variance.cpu().numpy().reshape(s)

        # plot raw, mean, and variance
        fig, ax = plt.subplots(3, sharex=True, sharey=True)
        cr = ax[0].scatter(inputs[:,0], inputs[:,1], c=targets, cmap='viridis', s=0.4, edgecolors='none')
        cm = ax[1].contourf(*inputsg, mean, levels=n_contours)
        cv = ax[2].contourf(*inputsg, variance, levels=n_contours)
        # indpts = self.variational_strategy.inducing_points.data.cpu().numpy()
        # indpts = self.cov._inducing_inv_root.
        # ax[2].plot(indpts[:,0], indpts[:,1], 'ko', markersize=3, alpha=0.2)

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


class SVGP(VariationalGP):

    def __init__(self, n_inducing):

        # number of inducing points and optimisation samples
        assert isinstance(n_inducing, int)
        self.m = n_inducing

        # variational distribution and strategy
        # NOTE: we put random normal dumby inducing points
        # here, which we'll change in self.fit
        vardist = CholeskyVariationalDistribution(self.m)
        varstra = VariationalStrategy(
            self,
            torch.randn((self.m, 2)),
            vardist,
            learn_inducing_locations=True
        )
        VariationalGP.__init__(self, varstra)

        # kernel — implemented in self.forward
        self.mean = ConstantMean()
        self.cov = MaternKernel(ard_num_dims=2)
        # self.cov = GaussianSymmetrizedKLKernel()
        self.cov = ScaleKernel(self.cov, ard_num_dims=2)
        
        # likelihood
        self.likelihood = GaussianLikelihood()

        # hardware allocation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.likelihood.to(self.device).float()
        self.to(self.device).float()

    def forward(self, input):
        m = self.mean(input)
        v = self.cov(input)
        return MultivariateNormal(m, v)

    def fit(self, inputs, targets, covariances=None, n_samples=5000, max_iter=10000, 
            learning_rate=1e-3, rtol=1e-4, n_window=100, auto=True, verbose=True):

        '''
        Optimises the hyperparameters of the GP kernel and likelihood.
        inputs: (nx2) numpy array
        targets: (n,) numpy array
        n_samples: number of samples to take from the inputs/targets at every optimisation epoch
        max_iter: maximum number of optimisation epochs
        learning_rate: optimiser step size
        rtol: change between -MLL values over ntol epoch that determine termination if auto==True
        ntol: number of epochs required to maintain rtol in order to terminate if auto==True
        auto: if True terminate based on rtol and ntol, else terminate at max_iter
        verbose: if True show progress bar, else nothing
        '''

        # inducing points randomly distributed over data
        indpts = np.random.choice(inputs.shape[0], self.m, replace=True)
        self.variational_strategy.inducing_points.data = torch.from_numpy(inputs[indpts]).to(self.device).float()

        # number of random samples
        n = inputs.shape[0]
        n = n_samples if n >= n_samples else n

        # objective
        mll = VariationalELBO(self.likelihood, self, n, combine_terms=True)

        # stochastic optimiser
        opt = torch.optim.Adam(self.parameters(),lr=learning_rate)
        # opt = torch.optim.SGD(self.parameters(),lr=learning_rate)

        # convergence criterion
        # print("N window ", n_window)
        if auto: criterion = ExpMAStoppingCriterion(rel_tol=rtol, minimize=True, n_window=n_window)

        # episode iteratior
        epochs = range(max_iter)
        epochs = tqdm.tqdm(epochs) if verbose else epochs

        # train
        self.train()
        self.likelihood.train()
        self.loss = list()
        for _ in epochs:

            # randomly sample from the dataset
            idx = np.random.choice(inputs.shape[0], n, replace=False)
            
            ## UI covariance
            if covariances is None or covariances.shape[1] == 2:
                input = torch.from_numpy(inputs[idx]).to(self.device).float()
                target = torch.from_numpy(targets[idx]).to(self.device).float()

            # if the inputs are distributional, sample them
            if covariances is not None:
                covariance = torch.from_numpy(covariances[idx]).to(self.device).float()
                input = MultivariateNormal(input, covariance).rsample()

            # compute loss, compute gradient, and update
            loss = -mll(self(input), target)
            opt.zero_grad()
            loss.backward()
            opt.step()

            # verbosity and convergence check
            if verbose:
                epochs.set_description('Loss {:.4f}'.format(loss.item()))
                self.loss.append(loss.detach().cpu().numpy())
            if auto and criterion.evaluate(loss.detach()):
                break

    def sample(self, x):

        '''
        Samples the posterior at x
        x: (n,2) numpy array
        returns:
            mu: (n,) numpy array of predictive mean at x
            sigma: (n,) numpy array of predictive variance at x
        '''

        ## On your source code, call:
        # self.likelihood.eval()
        # self.eval()
        ## before using this function to toggle evaluation mode

        # sanity
        assert len(x.shape) == x.shape[1] == 2

        # sample posterior
        # TODO: fast_pred_var activates LOVE. Test performance on PF
        # https://towardsdatascience.com/gaussian-process-regression-using-gpytorch-2c174286f9cc
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x = torch.from_numpy(x).to(self.device).float()
            dist = self.likelihood(self(x))
            return dist.mean.cpu().numpy(), dist.variance.cpu().numpy()

    def save_posterior(self, n, xlb, xub, ylb, yub, fname, verbose=True):

        '''
        Samples the GP posterior on a inform grid over the
        rectangular region defined by (xlb, xub) and (ylb, yub)
        and saves it as a pointcloud array.

        n: determines n² number of sampling locations
        xlb, xub: lower and upper bounds of x sampling locations
        ylb, yub: lower and upper bounds of y sampling locations
        fname: path to save array at (use .npy extension)
        '''

        # sanity
        assert('.npy' in fname)

        # toggle evaluation mode
        self.likelihood.eval()
        self.eval()
        torch.cuda.empty_cache()

        # posterior sampling locations
        inputs = [
            np.linspace(xlb, xub, n),
            np.linspace(ylb, yub, n)
        ]
        inputs = np.meshgrid(*inputs)
        inputs = [_.flatten() for _ in inputs]
        inputs = np.vstack(inputs).transpose()

        # split the array into smaller ones for memory
        inputs = np.split(inputs, 4000, axis=0)

        # compute the posterior for each batch
        means, variances = list(), list()
        with torch.no_grad():
            for i, input in enumerate(inputs):
                if verbose: print('Batch {}'.format(i))
                mean, variance = self.sample(input)
                means.append(mean)
                variances.append(variance)

        # assemble probabalistic pointcloud
        cloud = np.hstack((
            np.vstack(inputs),
            np.hstack(means).reshape(-1, 1),
            np.hstack(variances).reshape(-1, 1)
        ))

        # save it
        np.save(fname, cloud)

    def plot(self, inputs, targets, fname, n=80, n_contours=50, track=None):

        '''
        Plots:
            ax[0]: raw inputs and targets,
            ax[1]: posterior predictive mean,
            ax[2]: posterior predictive variance
        inputs: (n,2) numpy array of inputs
        output: (n,) numpy array of targets
        fname: path to save plot at (extension determines file type, e.g. .png or .pdf)
        n: determines n² number of sampling locations to plot GP posterior
        n_contours: number of contours to show output magnitude with
        '''

        # sanity
        assert inputs.shape[0] == targets.shape[0]
        assert inputs.shape[1] == 2

        # toggle evaluation mode
        self.likelihood.eval()
        self.eval()
        torch.cuda.empty_cache()

        # posterior sampling locations
        inputsg = [
            np.linspace(min(inputs[:,0]), max(inputs[:,0]), n),
            np.linspace(min(inputs[:,1]), max(inputs[:,1]), n)
        ]
        inputst = np.meshgrid(*inputsg)
        s = inputst[0].shape
        inputst = [_.flatten() for _ in inputst]
        inputst = np.vstack(inputst).transpose()
        inputst = torch.from_numpy(inputst).to(self.device).float()

        # sample
        with torch.no_grad():
            outputs = self(inputst)
            outputs = self.likelihood(outputs)
            mean = outputs.mean.cpu().numpy().reshape(s)
            variance = outputs.variance.cpu().numpy().reshape(s)

        # plot raw, mean, and variance
        fig, ax = plt.subplots(3, sharex=True, sharey=True)
        cr = ax[0].scatter(inputs[:,0], inputs[:,1], c=targets, cmap='jet', s=0.4, edgecolors='none')
        cm = ax[1].contourf(*inputsg, mean, cmap='jet', levels=n_contours)
        cv = ax[2].contourf(*inputsg, variance, levels=n_contours)
        indpts = self.variational_strategy.inducing_points.data.cpu().numpy()
        ax[2].plot(indpts[:,0], indpts[:,1], 'ko', markersize=1, alpha=0.2)

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

        if track != None:
            ax[0].plot(track[:,0], track[:,1], "-b", linewidth=0.2)

        # save
        fig.savefig(fname, bbox_inches='tight', dpi=1000)

    def plot_loss(self, fname):

        # plot
        fig, ax = plt.subplots(1)
        ax.plot(self.loss, 'k-')

        # format
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ELBO')
        ax.set_yscale('log')
        plt.tight_layout()

        # save
        fig.savefig(fname, bbox_inches='tight', dpi=1000)
        
    def save(self, fname):
        torch.save(self.state_dict(), fname)

    @classmethod
    def load(cls, nind, fname):
        gp = cls(nind)
        gp.load_state_dict(torch.load(fname))
        return gp

