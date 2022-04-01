#!/usr/bin/env python3

import torch, numpy as np, tqdm, matplotlib.pyplot as plt
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
from gp_mapping.convergence import ExpMAStoppingCriterion
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

from slam_msgs.msg import PlotPosteriorResult, PlotPosteriorAction
from slam_msgs.msg import SamplePosteriorResult, SamplePosteriorAction
from slam_msgs.msg import MinibatchTrainingAction, MinibatchTrainingResult, MinibatchTrainingGoal

import actionlib

import numpy as np

class SVGP_Particle(VariationalGP):

    def __init__(self, n_inducing):

        # number of inducing points and optimisation samples
        assert isinstance(n_inducing, int)
        self.s = n_inducing

        # variational distribution and strategy
        # NOTE: we put random normal dumby inducing points
        # here, which we'll change in self.fit
        vardist = CholeskyVariationalDistribution(self.s)
        varstra = VariationalStrategy(
            self,
            torch.randn((self.s, 2)),
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
        # TODO: optimize device allocation
        # with least_used_cuda_device():
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.likelihood.to(self.device).float()
        self.to(self.device).float()

        ####### ROS part
        self.storage_path = rospy.get_param("~results_path")
        self.count_training = 0
        
        self.node_name = rospy.get_name()
        self.particle_number = int(self.node_name.split('_')[1])

        # Subscription to GP inducing points from RBPF
        ip_top = rospy.get_param("~inducing_points_top")
        rospy.Subscriber(ip_top, PointCloud2, self.ip_cb, queue_size=1)
        self.inducing_points_received = False

        # AS for plotting results
        plot_gp_name = rospy.get_param("~plot_gp_server")
        self._as_plot = actionlib.SimpleActionServer(self.node_name + plot_gp_name, PlotPosteriorAction, 
                                                execute_cb=self.plot_posterior, auto_start = False)
        self._as_plot.start()

        # AS for expected meas
        sample_gp_name = rospy.get_param("~sample_gp_server")
        self._as_sample = actionlib.SimpleActionServer(self.node_name + sample_gp_name, SamplePosteriorAction, 
                                                execute_cb=self.sample_posterior, auto_start = False)
        self._as_sample.start()

        self.training = False
        self.plotting = False

        # AS for minibath training data from RBPF
        mb_gp_name = rospy.get_param("~minibatch_gp_server")
        self.ac_mb = actionlib.SimpleActionClient(mb_gp_name, MinibatchTrainingAction)
        self.ac_mb.wait_for_server()

        print("Particle ", self.particle_number, " instantiated")
        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(self.device)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(self.device)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(self.device)/1024/1024/1024))
        # rospy.spin()

    def forward(self, input):
        m = self.mean(input)
        v = self.cov(input)
        return MultivariateNormal(m, v)

    def ip_cb(self, ip_cloud):
        print("Inducing points received")

        # Store beams as array of 3D points
        wp_locations = []
        for p in pc2.read_points(ip_cloud, 
                                field_names = ("x", "y", "z"), skip_nans=True):
            wp_locations.append(p)
        wp_locations = np.asarray(wp_locations)
        wp_locations = np.reshape(wp_locations, (-1,3))

        # Inducing points distributed on grid over survey area
        ip_locations = [
            np.linspace(min(wp_locations[:,0]), max(wp_locations[:,0]), round(np.sqrt(self.s))),
            np.linspace(min(wp_locations[:,1]), max(wp_locations[:,1]), round(np.sqrt(self.s)))
        ]
        inputst = np.meshgrid(*ip_locations)
        inputst = [_.flatten() for _ in inputst]
        inputst = np.vstack(inputst).transpose()
        self.variational_strategy.inducing_points.data = torch.from_numpy(inputst[:, 0:2]).to(self.device).float()

        self.inducing_points_received = True
        print("Inducing points received")


    def train_map(self, mb_size=5000, learning_rate=1e-3, 
                  rtol=1e-4, n_window=100, auto=True, verbose=True):

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

        # Wait for the inducing points from the RBPF node
        while not rospy.is_shutdown() and not self.inducing_points_received:
            print("Particle ", self.particle_number, " waiting for inducing points")
            rospy.sleep(1)

        # Get beams for minibatch training as pcl
        goal = MinibatchTrainingGoal()
        goal.particle_id = self.particle_number
        goal.mb_size = mb_size
        self.ac_mb.send_goal(goal)
        self.ac_mb.wait_for_result()
        result = self.ac_mb.get_result()

        # Wait for server to have enough pings to start training
        while not rospy.is_shutdown() and not result.success:
            self.ac_mb.send_goal(goal)
            self.ac_mb.wait_for_result()
            result = self.ac_mb.get_result()
            print("Particle ", self.particle_number, " w waiting for MB data to be available")
            rospy.sleep(1)

        # Store beams as array of 3D points
        beams = []
        for p in pc2.read_points(result.minibatch, 
                                field_names = ("x", "y", "z"), skip_nans=True):
            beams.append(p)
        beams = np.asarray(beams)
        beams = np.reshape(beams, (-1,3))

        # number of random samples
        n = beams.shape[0]
        n = mb_size if n >= mb_size else n
        print("Particle ", self.particle_number, " starting the training")

        # objective
        mll = VariationalELBO(self.likelihood, self, n, combine_terms=True)

        # stochastic optimiser
        opt = torch.optim.Adam(self.parameters(),lr=learning_rate)
        # opt = torch.optim.SGD(self.parameters(),lr=learning_rate)

        # convergence criterion
        if auto: criterion = ExpMAStoppingCriterion(rel_tol=rtol, 
                                                    minimize=True, n_window=n_window)

        # # episode iteratior
        # epochs = range(max_iter)
        # epochs = tqdm.tqdm(epochs) if verbose else epochs

        # train
        self.train()
        self.likelihood.train()
        self.loss = list()
        # for _ in epochs:
        self.iterations = 0
        while not rospy.is_shutdown() and not self.plotting:

            # Get beams for minibatch training as pcl
            goal = MinibatchTrainingGoal()
            goal.particle_id = self.particle_number
            goal.mb_size = n
            self.ac_mb.send_goal(goal)
            self.ac_mb.wait_for_result()
            result = self.ac_mb.get_result()

            # If minibatch received from server
            if result.success:
                # Store beams as array of 3D points
                beams = []
                for p in pc2.read_points(result.minibatch, 
                                        field_names = ("x", "y", "z"), skip_nans=True):
                    beams.append(p)
                beams = np.asarray(beams)
                beams = np.reshape(beams, (-1,3))

                self.training = True
                input = torch.from_numpy(beams[:, 0:2]).to(self.device).float()
                target = torch.from_numpy(beams[:,2]).to(self.device).float()

                # indpts = np.random.choice(beams.shape[0], self.s, replace=False)        
                # self.variational_strategy.inducing_points.data = torch.from_numpy(beams[indpts, 0:2]).to(self.device).float()

                # compute loss, compute gradient, and update
                loss = -mll(self(input), target)
                opt.zero_grad()
                loss.backward()
                opt.step()
                # print("GP ", self.particle_number, " training iteration ", self.iterations)
                self.training = False

                # verbosity and convergence check
                # if verbose:
                    # epochs.set_description('Loss {:.4f}'.format(loss.item()))
                self.loss.append(loss.detach().cpu().numpy())
                if auto and criterion.evaluate(loss.detach()):
                    break

                self.iterations += 1
            
            else:
                rospy.sleep(1)
            
        print("Done with the training ", self.particle_number)
        print("GP ", self.particle_number, " training iteration ", self.iterations)

    def sample_posterior(self, goal):

        beams = []
        for p in pc2.read_points(goal.ping, 
                                field_names = ("x", "y", "z"), skip_nans=True):
            beams.append(p)

        beams = np.asarray(beams)
        beams = np.reshape(beams, (-1,3)) 

        while self.training:
            rospy.Rate(1).sleep()
            print("GP ", self.particle_number, " waiting for training before sampling")

        mu, sigma = self.sample(np.asarray(beams)[:, 0:2])

        # Set action as success
        result = SamplePosteriorResult()
        result.mu = mu
        result.sigma = sigma
        self._as_sample.set_succeeded(result)
        # self.plotting = False
        print("GP ", self.particle_number, " sampled")


    def plot_posterior(self, goal):

        # Flag to stop training
        self.plotting = True

        # Wait for training to stop
        while self.training:
            rospy.Rate(1).sleep()
            print("GP ", self.particle_number, " waiting for training before plotting")

        beams = []
        for p in pc2.read_points(goal.pings, 
                                field_names = ("x", "y", "z"), skip_nans=True):
            beams.append(p)
        beams = np.asarray(beams)
        beams = np.reshape(beams, (-1,3)) 

        # Plot posterior and save it to image
        print("Plotting GP ", self.particle_number)
        self.plotting = True
        self.plot(beams[:,0:2], beams[:,2], 
                     self.storage_path + 'particle_' + str(self.particle_number) 
                     + '_training.png',
                     n=50, n_contours=100 )

        # Set action as success
        result = PlotPosteriorResult()
        result.success = True
        self._as_plot.set_succeeded(result)
        # self.plotting = False
        print("GP posterior plotted ", self.particle_number)


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

    def plot(self, inputs, targets, fname, n=80, n_contours=50):

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
        cr = ax[0].scatter(inputs[:,0], inputs[:,1], c=targets, cmap='viridis', s=0.4, edgecolors='none')
        cm = ax[1].contourf(*inputsg, mean, levels=n_contours)
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

        # save
        fig.savefig(fname, bbox_inches='tight', dpi=1000)

        self.plot_loss(self.storage_path + 'particle_' + str(self.particle_number) 
                        + '_training_loss.png')

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

if __name__ == '__main__':

    rospy.init_node('rbpf_svgp' , disable_signals=False)

    try:
        num_inducing_points = rospy.get_param("~num_inducing_points", 100)
        particle_map1 = SVGP_Particle(num_inducing_points)
        particle_map2 = SVGP_Particle(num_inducing_points)
        particle_map3 = SVGP_Particle(num_inducing_points)
        particle_map4 = SVGP_Particle(num_inducing_points)
        particle_map5 = SVGP_Particle(num_inducing_points)
        particle_map6 = SVGP_Particle(num_inducing_points)
        particle_map7 = SVGP_Particle(num_inducing_points)
        particle_map8 = SVGP_Particle(num_inducing_points)
        particle_map9 = SVGP_Particle(num_inducing_points)
        particle_map10 = SVGP_Particle(num_inducing_points)
        # particle_map.train_map(mb_size=1000, learning_rate=1e-1, 
        #                        rtol=1e-4, n_window=100, auto=False, verbose=True)
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.logerr("Couldn't launch rbpf_svgp")