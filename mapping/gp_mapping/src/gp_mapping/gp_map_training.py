#!/usr/bin/env python3

# from process import process
import os
from gp import SVGP
import numpy as np
from optparse import OptionParser
import numpy as np


def train_svgp(gp_inputs_type, survey_name):

    print("Loading ", survey_name)
    cloud = np.load(survey_name)
    points = cloud['points']
    inputs = points[:, [0,1]]
    print("Inputs ", inputs.shape)
    targets = points[:,2]
    print("Targets ", targets.shape)
    
    print(gp_inputs_type)
    if gp_inputs_type == 'di':
        name = "svgp_di"
        covariances = None
    else:
        ## UI
        covariances = cloud['covs'][:,0:2,0:2]
        print("Covariances ", covariances.shape)
        name = "svgp_ui"    

    # initialise GP with 1000 inducing points
    gp = SVGP(1000)
    gp.fit(inputs, targets, covariances=covariances, n_samples=4000, 
            max_iter=1000, learning_rate=1e-1, rtol=1e-12, n_window=2000, 
            auto=False, verbose=True)
   
    # Save GP
    print("Saving trained GP")
    gp.save(name + '.pth')

    # save figures
    print("Plotting results")
    gp.plot(inputs, targets, name + '.png',
             n=100, n_contours=100)
    gp.plot_loss(name + '_loss.png')
    
    # Save loss for tunning of stopping criterion
    np.save(name + '_loss.npy', np.asarray(gp.loss))

    # Save posterior
    print("Saving posterior")
    x = inputs[:,0]
    y = inputs[:,1]
    gp.save_posterior(1000, min(x), max(x), min(y), max(y), 
                      name + '_post.npy', verbose=False)


def trace_kernel(gp_path):

    gp = SVGP.load(1000, gp_path)
    gp.likelihood.eval()
    gp.eval()

    print("Kernel")
    ip = gp.variational_strategy.inducing_points.data
    print(np.trace(gp.cov(ip).cpu().numpy()))


if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option("--gp_inputs", dest="gp_inputs",
                  default="", help="di or ui inputs for training.")
    parser.add_option("--survey_name", dest="survey_name",
                  default="", help="Name for folder to store results.")

    (options, args) = parser.parse_args()
    gp_inputs_type = options.gp_inputs
    survey_name = options.survey_name

    train_svgp(gp_inputs_type, survey_name)
    # trace_kernel(survey_name)
