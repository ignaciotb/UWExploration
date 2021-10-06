#!/usr/bin/env python3

#  from process import process
from gp import SVGP
import numpy as np
import matplotlib.pyplot as plt
from gpytorch.distributions import MultivariateNormal
import torch

def train_svgp():

    # # generate and save processed numpy pointcloud (n,3)
    # cloud = process(
    #     '../../data/KTH_Post_Deployment_AVG_WGS84UTM32N_RH200_50cm.xyz',
    #     '../../data/mbes_pings.cereal',
    #     '../../data/cloud.npy'
    # )
    # del cloud

    # load the generated pointcloud
    cloud = np.load('/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/utils/uw_tests/datasets/baggens_2021/pcl_baggens_2021.npy')
    inputs = cloud[:,[0,1]]
    targets = cloud[:,2]

    print(inputs[0])
    print(targets.shape)
    # initialise GP with 1000 inducing points
    gp = SVGP(1000)
    gp.fit(inputs, targets, n_samples=1000, max_iter=1000, learning_rate=1e-2, rtol=1e-4, ntol=100, auto=False, verbose=True)

    # save figure
    gp.plot(inputs, targets, '/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/utils/uw_tests/datasets/baggens_2021/svgp_1000_0.01.png', n=100, n_contours=100)

    # save the model
    gp.save('/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/utils/uw_tests/datasets/baggens_2021/svgp_1000_0.01.pth')

#  def example1():
#
    #  # load the generated pointcloud
    #  cloud = np.load('../../data/cloud.npy')
    #  inputs = cloud[:,[0,1]]
    #  targets = cloud[:,2]
#
    #  # load GP
    #  gp = SVGP.load(1000, '../../models/svgp.pth')
#
    #  # plot
    #  # gp.plot(inputs, targets, '../../img/plotn.png', n=100, n_contours=100)
#
    #  # sample the model
    #  # mu, sigma = gp.sample(inputs[:10])
    #  x = inputs[:,0]
    #  y = inputs[:,1]
#
    #  # print(gp.sample(inputs[:10]))
    #  gp.save_posterior(1000, min(x), max(x), min(y), max(y), '../../data/posterior.npy', verbose=True)
#
    #  cloud = np.load('../../data/posterior.npy')
    #  print(cloud.shape)
    #  # fig, ax = plt.subplots(2)
    #  # ax[0].scatter(cloud[:,0], cloud[:,1], c=cloud[:,2], cmap='viridis', s=0.4, edgecolors='none')
    #  # ax[1].scatter(cloud[:,0], cloud[:,1], c=cloud[:,3], cmap='viridis', s=0.4, edgecolors='none')
    #  # fig.savefig('yolo.png', dpi=500)

def plot_and_save_posterior():

    # load the generated pointcloud
    # cloud = np.load('../../data/lost_targets/cloud.npy')
    cloud = np.load('../../data/overnight_20/pcl_33_over.npy')
    inputs = cloud[:,:2]
    targets = cloud[:,2]

    # load pertaining GP
    gp = SVGP.load(1000, '../../models/overnight_20/svgp_1000_0.01.pth')

    # plot the data and posterior
    gp.plot(inputs, targets, '../../img/overnight_20/svgp_1000_0.01.png')

    # save the posterior point cloud
    x = inputs[:,0]
    y = inputs[:,1]
    gp.save_posterior(1000, min(x), max(x), min(y), max(y), '../../data/overnight_20/svgp_1000_0.01.npy', verbose=True)

    # # covariance matricies: shape == (n,2,2)
    # cov = np.eye(2)*0.001**2
    # cov = np.broadcast_to(cov, (inputs.shape[0], 2, 2))



    # # instantiate SVGP
    # gp = SVGP(1000)

    # # train
    # gp.fit(
    #     inputs, 
    #     targets,
    #     covariances=None,
    #     n_samples=8000, 
    #     max_iter=10000, 
    #     learning_rate=1e-2, 
    #     rtol=1e-6, 
    #     ntol=200, 
    #     auto=False, 
    #     verbose=True
    # )

    # # save
    # gp.save('../../models/pcl.pth')

    # # plot posterior
    # gp.plot(inputs, targets, '../../img/pcl.png', n=100, n_contours=50)




if __name__ == '__main__':
    #  plot_and_save_posterior()
    train_svgp()

