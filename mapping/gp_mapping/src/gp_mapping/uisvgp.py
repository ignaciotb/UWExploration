#!/usr/bin/env python3

from typing import Tuple

import numpy as np
import tensorflow as tf

from gpflow import kullback_leiblers
from gpflow.base import Parameter
from gpflow.conditionals import conditional
from gpflow.config import default_float
from gpflow.utilities import positive, triangular
from gpflow.models.model import GPModel, InputData, MeanAndVariance, RegressionData
from gpflow.models.training_mixins import ExternalDataTrainingLossMixin
from gpflow.models.util import inducingpoint_wrapper

class UISVGP(GPModel, ExternalDataTrainingLossMixin):
    """
    This is the Uncertain Input Sparse Variational GP (SVGP). The key reference is
    ::
      @inproceedings{
          fsdfsd
      }
    """

    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        *,
        mean_function=None,
        num_latent_gps: int = 1,
        num_input_dims: int = 2,
        q_diag: bool = False,
        q_f_mu=None,
        q_f_sqrt=None,
        q_x_mu=None,
        q_x_sqrt=None,
        whiten: bool = True,
        num_data=None,
    ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_data = num_data
        self.q_diag = q_diag
        self.whiten = whiten
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        # init variational parameters
        num_inducing = self.inducing_variable.num_inducing
        self.num_input_dims = num_input_dims
        self._init_variational_parameters(num_inducing, q_f_mu, q_f_sqrt, q_x_mu, q_x_sqrt, q_diag)


    def _init_variational_parameters(self, num_inducing, q_f_mu, q_f_sqrt, q_x_mu, q_x_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_f_mu` and `q_f_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_f_mu` and `q_f_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.
        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.
        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
            This shall be the same for the input and output distributions, q(x) & q(u).
        :param q_f_mu: np.array or None
            Mean of the variational Gaussian posterior over the outputs. 
            If None the function will initialise the mean with zeros. 
            If not None, the shape of `q_f_mu` is checked.
        :param q_f_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior over the inputs.
            If None the function will initialise `q_f_sqrt` with identity matrix.
            If not None, the shape of `q_f_sqrt` is checked, depending on `q_diag`.
        :param q_x_mu: np.array or None
            Mean of the variational Gaussian posterior over the inputs. 
            If None the function will initialise the mean with zeros. 
            If not None, the shape of `q_x_mu` is checked.
        :param q_x_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior over the outputs.
            If None the function will initialise `q_x_sqrt` with identity matrix.
            If not None, the shape of `q_x_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_f_mu`, `q_f_sqrt`, `q_x_mu`, and `q_x_sqrt` have 
            the correct shape or to construct them with the correct shape. 
            If `q_diag` is true, `q_f_sqrt` and `q_x_sqrt` is two dimensional and 
            only holds the square root of the covariance diagonal elements. 
            If False, `q_f_sqrt` and `q_x_sqrt` är three dimensional.
        """

        '''
        Chris notes:
        In our case, self.num_latent_gps shall be 1 for the seafloor elevation.
        Hence, q_x_mu should perhaps be initialised with shape (num_inducing, D).
        '''

        # mean of posterior over outputs q(u) = q(f(z))
        q_f_mu = np.zeros((num_inducing, self.num_latent_gps)) if q_f_mu is None else q_f_mu
        self.q_f_mu = Parameter(q_f_mu, dtype=default_float())  # [M, P]
        
        # covariance of posterior over outputs q(u) = q(f(z))
        if q_f_sqrt is None:
            if self.q_diag:
                ones = np.ones((num_inducing, self.num_latent_gps), dtype=default_float())
                self.q_f_sqrt = Parameter(ones, transform=positive())  # [M, P]
            else:
                q_f_sqrt = [
                    np.eye(num_inducing, dtype=default_float()) for _ in range(self.num_latent_gps)
                ]
                q_f_sqrt = np.array(q_f_sqrt)
                self.q_f_sqrt = Parameter(q_f_sqrt, transform=triangular())  # [P, M, M]
        else:
            if q_diag:
                assert q_f_sqrt.ndim == 2
                self.num_latent_gps = q_f_sqrt.shape[1]
                self.q_f_sqrt = Parameter(q_f_sqrt, transform=positive())  # [M, L|P]
            else:
                assert q_f_sqrt.ndim == 3
                self.num_latent_gps = q_f_sqrt.shape[0]
                num_inducing = q_f_sqrt.shape[1]
                self.q_f_sqrt = Parameter(q_f_sqrt, transform=triangular())  # [L|P, M, M]

                
        # mean of posterior over inputs q(X) [M, D]
        q_x_mu = np.zeros((num_inducing, self.num_input_dims)) if q_x_mu is None else q_x_mu
        self.q_x_mu = Parameter(q_x_mu, dtype=default_float())  # [M, D]
        
        # covariance of posterior over outputs q(u) = q(f(z)) [L, D*M, D*M]
        if q_x_sqrt is None:
            if self.q_diag:
                ones = np.ones((num_inducing*self.num_input_dims, self.num_latent_gps), dtype=default_float())
                self.q_x_sqrt = Parameter(ones, transform=positive())  # [M*D, L]
            else:
                # q_x_sqrt = [
                #     np.eye(num_inducing*self.num_input_dims, dtype=default_float()) for _ in range(self.num_latent_gps)
                # ]
                q_x_sqrt = np.eye(num_inducing*self.num_input_dims, dtype=default_float())
                # q_x_sqrt = np.array(q_x_sqrt)
                self.q_x_sqrt = Parameter(q_x_sqrt, transform=triangular())  # [L, D*M, D*M]
        else:
            if q_diag:
                assert q_x_sqrt.ndim == 2
                # self.num_latent_gps = q_x_sqrt.shape[1]
                self.q_x_sqrt = Parameter(q_x_sqrt, transform=positive())  # [M*D, L]
            else:
                assert q_x_sqrt.ndim == 3
                # self.num_latent_gps = q_x_sqrt.shape[0]
                # num_inducing = q_x_sqrt.shape[1]
                self.q_x_sqrt = Parameter(q_x_sqrt, transform=triangular())  # [L, D*M, D*M]
            
    # KL[q(f)||p(f)]
    def prior_kl_f(self) -> tf.Tensor:
        return kullback_leiblers.prior_kl(
            self.inducing_variable, self.kernel, self.q_f_mu, self.q_f_sqrt, whiten=self.whiten
        )

    # We'll have a second kl term here
    # for KL[q(x)||p(x)]
    def prior_kl_x(self, Xs_mean, Xs_cov) -> tf.Tensor:
        # TODO: create new kullback_leiblers.prior_kl() for our case
        return kullback_leiblers.prior_kl_x(
            self.inducing_variable, Xs_mean, Xs_cov, self.q_x_mu, self.q_x_sqrt, whiten=self.whiten
        )

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo_ui(data)
    
    def elbo_ui(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """

        # Our data now should contain X_mean and X_var for UIs
        Xn, Yn, Xs_mean, Xs_cov = data

        # The two KL terms
        kl_f = self.prior_kl_f()
        kl_x = self.prior_kl_x(Xs_mean, Xs_cov)
        
        # Calculate f(x_n) for all the input points
        f_mean, f_var = self.predict_f(Xn, full_cov=False, full_output_cov=False)
        
        # The two terms of the loglikelihood log(p(z|f,x))
        var_exp_f = self.likelihood.variational_expectations(f_mean, f_var, Yn)
        #  var_exp_x = self.likelihood.variational_expectations(X_mean, X_var, Z)
        
        # The scale term
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl_f.dtype)
            minibatch_size = tf.cast(tf.shape(Xn)[0], kl_f.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl_f.dtype)

        # Our new elbo
        return tf.reduce_sum(var_exp_f) * scale - kl_f - kl_x

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """

        X, Y = data
        kl = self.prior_kl_f()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        q_mu = self.q_f_mu
        q_sqrt = self.q_f_sqrt
        mu, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        return mu + self.mean_function(Xnew), var