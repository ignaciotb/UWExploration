
from gpflow.models import SVGP
from gpflow.likelihoods import Gaussian, Likelihood
from gpflow.kernels import Matern12, Kernel
from gpflow.ci_utils import ci_niter
from gpflow import set_trainable

import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import pathlib
import tempfile

from uisvgp import UISVGP


class Problem:

    def __init__(self, X, Y, m, X_cov=None, kernel=Matern12()):
        '''
        This class represents the base class for the general class
        of regression problems we want to solve.

        We consider three regression methods:
        1. SVGP regression with deterministic inputs (`di`).
        2. SVGP regression with uncertain independent inputs (`uii`),
        3. SVGP regression with uncertain correlated inputs (`uci`).

        :note:
            Method `'uci'` requires a special model `UISVGP`.

        Parameters
        ----------
        :param X: np.array
            Input data array of shape `(n,Dx)`. 
            Considered the the mean values if `Xvar` is also given.
        :params Y: np.array
            Output data array of shape `(n,Dy)`, i.e. f + noise.
        :params m: int
            Number of inducing variables `Z`.
            Also considered to be the number of samples 
        :params method: str
            Regression methods:
            1. `'di'`,
            2. `'uii'`,
            3. `'uci'`.
        '''

        # sanity
        assert X.shape[0] >= 1
        assert X.shape[0] == Y.shape[0]
        assert len(X.shape) == len(Y.shape) == 2

        # number of raw data points
        self.n = X.shape[0]

        # dimensions of inputs and outputs
        self.dx = X.shape[1]
        self.dy = Y.shape[1]

        # data
        self.X = X
        self.Y = Y
        self.X_cov = X_cov

        # loss record
        self.loss = list()

        # determinsitic or uncertain-uncorrelated inputs
        if self.X_cov is None or self.X_cov.shape == (self.n, self.dx, self.dx):

            # inducing location indicies
            if isinstance(m, int):
                self.u_idx = np.random.choice(self.n, m, replace=False)
                self.m = m
            elif len(m.shape) == 1:
                self.u_idx = m
                self.m = m.shape[0]
            else:
                raise ValueError('m must be integer or 1D array of indicies.')

            # vanilla SVGP
            self.model = SVGP(
                kernel,
                Gaussian(),
                self.X[self.u_idx].copy(),
                num_data=self.n
            )

        # covariance matrix at inducing locations, given with indicies
        elif self.X_cov.shape == (self.dx*m.shape[0], self.dx*m.shape[0]):

            # inducing location indicies
            self.u_idx = m
            self.m = m.shape[0]

            # UCISVGP
            self.model = UISVGP(
                kernel,
                Gaussian(),
                self.X[self.u_idx].copy(),
                num_data=self.n
            )
            set_trainable(self.model.inducing_variable, False)
            print('Assuming uncertain correlated inputs.')

        # feedback
        else:
            raise ValueError('Invalid X_cov and m shape.')

    @staticmethod
    def sample_uci(X_mean, X_cov, Z):
        # print(X_mean.shape, X_cov.shape, Z.shape)

        d = np.size(X_mean, 1)  # Dim of latent variables P

        # Allocate output
        Xs_mean = np.zeros((len(Z), d))
        Xs_cov = np.zeros((len(Z)*d, len(Z)*d))

        # TODO: try to get rid of nested for loops
        ## Covariance
        Xs_mean = X_mean[Z]
        cnt = 0
        for i in Z:
            # Mean
            # Cov Diagonal terms
            Xs_cov[cnt*d:cnt*d+d, cnt*d:cnt*d +
                   d] = X_cov[i*d:i*d + d, i*d:i*d+d]
            # Cov off-diagonal terms
            cnt_j = cnt+1
            for j in Z[cnt_j:]:
                Xs_cov[cnt*d:cnt*d+d, cnt_j*d:cnt_j *
                       d+d] = X_cov[i*d:i*d + d, j*d:j*d+d]
                Xs_cov[cnt_j*d:cnt_j*d+d, cnt*d:cnt*d +
                       d] = X_cov[i*d:i*d + d, j*d:j*d+d].T
                cnt_j += 1
            cnt += 1

        return (Xs_mean, Xs_cov)

    def optimise(self, batch_size=2000, epochs=1000, learning_rate=1e-2, verbose=True):

        # minibatch iterator over raw data
        batches = iter(
            tf.data.Dataset.from_tensor_slices(
                (self.X, self.X_cov, self.Y)
                #  if self.X_cov.shape == (self.n, self.dx, self.dx)
                if self.X_cov is not None
                else (self.X, self.Y)
            )
            .prefetch(tf.data.experimental.AUTOTUNE)
            .repeat()
            .shuffle(self.X.shape[0])
            .batch(batch_size)
        )

        # sample covariance with inducing points
        # Xs_mean, Xs_cov = self.sample_uci(self.X, self.X_cov, self.X_indicies)

        # optimisation algorithm
        optimiser = tf.optimizers.Adam(learning_rate=learning_rate)

        # compiled optimisation stepping function
        @tf.function
        def optimisation_step(batch):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.model.trainable_variables)

                # determinstic inputs
                if self.X_cov is None:
                    X, Y = batch
                    loss = self.model.training_loss((X, Y))
                    #  raise NotImplementedError('Coming soon.')

                # uncertain uncorrelated inputs
                elif self.X_cov.shape == (self.n, self.dx, self.dx):

                    # unpack batch
                    X, X_var, Y = batch
                    X_dist = tfp.distributions.MultivariateNormalFullCovariance(
                        loc=X, covariance_matrix=X_var
                    )

                    # compute loss
                    loss = self.model.training_loss((X_dist.sample(), Y))

                # uncertain correlated inputs
                elif self.X_cov.shape == (self.dx*self.m, self.dx*self.m):

                    # loss
                    # NOTE: self.X[self.u_idx] is the same thing as self.model.inducing_variable.Z
                    loss = self.model.training_loss(
                        (*batch, self.X[self.u_idx], self.X_cov))

            grads = tape.gradient(loss, self.model.trainable_variables)
            optimiser.apply_gradients(
                zip(grads, self.model.trainable_variables))
            return loss

        # training
        epochs = trange(epochs) if verbose else range(epochs)
        for epoch in epochs:
            loss = optimisation_step(next(batches))
            if verbose:
                epochs.set_description('Loss: {}'.format(loss.numpy()))
                self.loss.append(-loss.numpy())

    def plot_loss(self, fname):

        # plot
        fig, ax = plt.subplots(1)
        ax.plot(self.loss, 'k-')

        # format
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ELBO')
        # ax.set_yscale('logit')
        plt.tight_layout()

        # save
        fig.savefig(fname, bbox_inches='tight', dpi=1000)

    def plot_prediction(self, n, fname):
        raise NotImplementedError('This must be defined in child class.')


class Scalar_Function(Problem):

    def __init__(self, X, Y, m, Xvar=None, method='uii'):

        # sanity
        assert(X.shape[1] == Y.shape[1] == 1)

        # inherit
        Problem.__init__(self, X, Y, m, Xvar=Xvar, method=method)

    def plot_prediction(self, n, fname):

        # predictions
        x = np.linspace(min(self.X[:, 0]), max(self.X[:, 0]), num=n)
        y_mu, y_sigma = self.model.predict_y(x.reshape(-1, 1))

        # plot
        fig, ax = plt.subplots(1)
        ax.plot(self.X[:, 0], self.Y[:, 0], 'k-', label='$y$')
        ax.plot(x, y_mu[:, 0], 'k--', label='$GP$')
        ax.fill_between(
            x,
            (y_mu-2*y_sigma**0.5)[:, 0],
            (y_mu+2*y_sigma**0.5)[:, 0],
            alpha=0.5,
            label='Standard deviation'
        )
        Z = self.model.inducing_variable.Z.numpy()
        ax.plot(Z, np.zeros_like(Z), 'k|', label='Inducing variables')
        ax.legend()

        # formatting
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        plt.tight_layout()

        # save
        fig.savefig(fname, bbox_inches='tight', dpi=1000)

    @classmethod
    def test(cls):

        # input mean and variance
        x_mu = np.linspace(-5, 5, num=1000).reshape(-1, 1)
        x_sigma = np.broadcast_to(np.eye(1)*0.001*2, (x_mu.shape[0], 1, 1))

        # outputs
        y = tf.math.sigmoid(
            x_mu) + np.random.normal(scale=0.001, size=x_mu.shape)

        # problem
        prob = cls(x_mu, y, 100, Xvar=x_sigma, method='uci')
        print(prob.model.inducing_variable.Z)
        # prob.optimise(batch_size=500, epochs=1000, learning_rate=1e-2, verbose=True)
        # prob.plot_prediction(1000, '../../img/test/scalar_function_prediction.png')
        # prob.plot_loss('../../img/test/scalar_function_loss.png')


class Scalar_Field(Problem):

    def __init__(self,  X, Y, m, X_cov=None):

        # sanity
        assert(X.shape[1] == 2 and Y.shape[1] == 1)

        # inherit
        Problem.__init__(self, X, Y, m, X_cov=X_cov)

    def plot_prediction(self, n, fname):

        # test locations on grid
        X_grid = [np.linspace(min(self.X[:, i]), max(self.X[:, i]), n)
                  for i in range(self.X.shape[1])]
        X_test = np.meshgrid(*X_grid)
        s = X_test[0].shape
        X_test = [_.flatten() for _ in X_test]
        X_test = np.vstack(X_test).transpose()

        # mean and variance at test locations
        mu, sigma = self.model.predict_y(X_test)
        mu, sigma = mu.numpy().reshape(s), sigma.numpy().reshape(s)

        # plots for raw, mean and variance
        fig, ax = plt.subplots(3, sharex=True, sharey=True)
        cr = ax[0].scatter(self.X[:, 0], self.X[:, 1], c=self.Y[:, 0],
                           cmap='viridis', s=0.4, edgecolors='none')
        cm = ax[1].contourf(*X_grid, mu, levels=50)
        cv = ax[2].contourf(*X_grid, sigma, levels=50)

        # plot inducing locations
        Z = self.model.inducing_variable.Z.numpy()
        ax[2].plot(Z[:, 0], Z[:, 1], 'ko', markersize=3, alpha=0.2)

        # equal aspect ratio and colorbars
        for i, c in enumerate([cr, cm, cv]):
            ax[i].set_aspect('equal')
            fig.colorbar(c, ax=ax[i])

        # extra formatting
        ax[0].set_title('Raw data')
        ax[0].set_ylabel('$y~[m]$')
        ax[1].set_title('Mean')
        ax[1].set_ylabel('$y~[m]$')
        ax[2].set_title('Variance')
        ax[2].set_xlabel('$x~[m]$')
        ax[2].set_ylabel('$y~[m]$')
        plt.tight_layout()

    def save_model(self, fname):

        # save model
        self.model.predict_y_compiled = tf.function(
            self.model.predict_y, input_signature=[
                tf.TensorSpec(shape=[None, 2], dtype=tf.float64)]
        )
        tf.saved_model.save(
            self.model, fname)

    @classmethod
    def test(cls):

        # 3D pointcloud
        data = np.load(
            '/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/utils/uw_tests/datasets/overnight_2020/pcl_33_over.npy')

        # surface location (X) and elevation (Y)
        n = data.shape[0]
        X, Y = data[:, :2], data[:, 2:]
        d = X.shape[1]

        # indicies upon which to place inducing points
        m = 1000  # number of inducing points
        u_idx = np.random.choice(n, m, replace=False)

        # prior on inducing point locations
        # X_cov = np.eye(2)*0.001**2
        # X_cov = np.broadcast_to(X_cov, (X.shape[0], 2, 2))
        #  X_cov = np.random.randn(m*d, m*d)*0.001
        #  X_cov = X_cov @ X_cov.T

        # initialise problem
        prob = cls(X, Y, u_idx)
        # Minibatch size, number of training rounds and learning rate
        prob.optimise(batch_size=1000, epochs=1000, learning_rate=1e-2)

        # plot
        prob.plot_prediction(
            300, '/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/utils/uw_tests/datasets/overnight_2020/scalar_field_prediction.png')
        prob.plot_loss(
            '/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/utils/uw_tests/datasets/overnight_2020/scalar_field_loss.png')
        prob.save_model(
            '/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/utils/uw_tests/datasets/overnight_2020/svgp')


if __name__ == '__main__':

    # Scalar_Function.test()
    Scalar_Field.test()
    # print(tf.config.list_physical_devices("GPU"))
