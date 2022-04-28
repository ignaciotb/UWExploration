import os, sys

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
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float32, Int32

from slam_msgs.msg import PlotPosteriorResult, PlotPosteriorAction
from slam_msgs.msg import SamplePosteriorResult, SamplePosteriorAction
from slam_msgs.msg import MinibatchTrainingAction, MinibatchTrainingResult, MinibatchTrainingGoal

import actionlib

import numpy as np

import warnings
import time
from pathlib import Path

from collections import OrderedDict


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


model = SVGP(10)

path = "/tmp/my.fifo"
try:
    fifo = open(path, "r")
except Exception as e:
    print (e)
    sys.exit()

import collections
import json
import ast

time_start = time.time()
with open(path, 'r') as fifo:
    #  while True:
        #  line = fifo.readline()[:-1]
    data = fifo.read()
    split_list = data.split(";")
    tensor_list = []
    for el in split_list[:-1]:
        if "inf" in el:
            el = el.replace("inf", "float(\"inf\")")
        tens = torch.tensor(eval(el[:-16]))
        tensor_list.append(tens.to(torch.device(el[-6:])))

    odict = OrderedDict([('variational_strategy.inducing_points',
                           tensor_list[0]),
                          ('variational_strategy.variational_params_initialized',
                           tensor_list[1]),
                          ('variational_strategy.updated_strategy',
                           tensor_list[2]),
                          ('variational_strategy._variational_distribution.variational_mean',
                           tensor_list[3]),
                          ('variational_strategy._variational_distribution.chol_variational_covar',
                           tensor_list[4]),
                          ('mean.constant',
                           tensor_list[5]),
                          ('cov.raw_outputscale',
                           tensor_list[6]),
                          ('cov.base_kernel.raw_lengthscale',
                           tensor_list[7]),
                          ('cov.base_kernel.raw_lengthscale_constraint.lower_bound',
                           tensor_list[8]),
                          ('cov.base_kernel.raw_lengthscale_constraint.upper_bound',
                           tensor_list[9]),
                          ('cov.raw_outputscale_constraint.lower_bound',
                           tensor_list[10]),
                          ('cov.raw_outputscale_constraint.upper_bound',
                           tensor_list[11])])

    print(odict)

    #  json_msg = json.loads(data[1:])
    #  json_acceptable_string = data.replace("'", "\"")
    #  print(type(collections.OrderedDict(json_msg)))
        #  if len(data) == 0:
            #  print ("Sender Terminated")
            #  break
    
    print("Time fifo ", time.time() - time_start)
fifo.close()

