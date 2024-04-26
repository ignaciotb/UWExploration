#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import filelock
import pickle
import rospy
import torch
from gpytorch.likelihoods import GaussianLikelihood
import time
from botorch.acquisition import UpperConfidenceBound
from AcquisitionFunctionClass import UCB_xy


class UpdateDist(object):
    """ Class that updates frame for real time animation plotting """
    
    def __init__(self, ax):
        """ Constructor

        Args:
            ax (ndarray): matplotlib.plt axis array, shape (2,2)
        """
        
        # Create file locks to interact safely with files
        # that might be concurrently written to
        self.lock_environment_gp = filelock.FileLock("GP_env.pickle.lock")
        self.lock_heading_gp     = filelock.FileLock("GP_angle.pickle.lock")

        # Set up 
        self.bounds = [-260, -40, 100, -70]
        self.ax = ax
        
    def __call__(self, j):
        """ animation function used by FuncAnimation to update drawings

        Args:
            j (any): dummy iteration variable, required by FuncAnimation signature

        Returns:
            iterable: list with zorder of contour path collections
        """
        
        # Clear axis
        plt.cla()
        #plt.clf()
        
        # Calculate time (during testing)
        t1 = time.time()
        
        # Load first model
        with self.lock_environment_gp:
            model1 = pickle.load(open("GP_env.pickle","rb"))
        model1.model.eval()
        model1.likelihood.eval()
        likelihood1 = GaussianLikelihood()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        likelihood1.to(device).float()
        model1.to(device).float()
        torch.cuda.empty_cache()
        
        
        # Plotting params
        n = 50
        n_contours = 25

        # posterior sampling locations for first GP
        inputsg = [
            np.linspace(self.bounds[0], self.bounds[1], n),
            np.linspace(self.bounds[3], self.bounds[2], n)
        ]
        inputst = np.meshgrid(*inputsg)
        s = inputst[0].shape
        inputst = [_.flatten() for _ in inputst]
        inputst = np.vstack(inputst).transpose()
        
        ucb_fun = UCB_xy(model1, beta=30)

        
        # Outputs for GP 1
        mean_list = []
        var_list = []
        ucb_list = []
        divs = 10
        with torch.no_grad():
            for i in range(0, divs):
                # sample
                inputst_temp = torch.from_numpy(inputst[i*int(n*n/divs):(i+1)*int(n*n/divs), :]).to(device).float()
                outputs = model1(inputst_temp)
                mean_r, sigma_r = ucb_fun._mean_and_sigma(inputst_temp)
                #ucb = ucb_fun.forward(inputst_temp.unsqueeze(-2))
                ucb = abs(mean_r - model1.model.mean_module.constant) + ucb_fun.beta.sqrt() * sigma_r
                outputs = likelihood1(outputs)
                mean_list.append(outputs.mean.cpu().numpy())
                var_list.append(outputs.variance.cpu().numpy())
                ucb_list.append(ucb.cpu().numpy())

        mean = np.vstack(mean_list).reshape(s)
        variance = np.vstack(var_list).reshape(s)
        ucb = np.vstack(ucb_list).reshape(s)
        points = model1.model.variational_strategy.inducing_points.detach().numpy()
        
        
        # Load second model
        with self.lock_heading_gp:
            model2 = pickle.load(open("GP_angle.pickle","rb"))
        model2.eval()
        model2.likelihood.eval()
        likelihood2 = GaussianLikelihood()
        likelihood2.to(device).float()
        model2.to(device).float()
        torch.cuda.empty_cache()

        samples1D = np.linspace(-np.pi, np.pi, n)

        # Outputs for GP 2
        mean_list = []
        var_list = []
        with torch.no_grad():
            inputst_temp = torch.from_numpy(samples1D).to(device).float()
            outputs = model2(inputst_temp)
            outputs = likelihood2(outputs)
            mean_list.append(outputs.mean.cpu().numpy())
            var_list.append(outputs.variance.cpu().numpy())

        mean2 = np.vstack(mean_list).squeeze(0)
        variance2 = np.vstack(var_list).squeeze(0)
        
        # Plots
        # [0, 0]: Mean of first GP
        # [0, 1]: Variance of first GP
        # [1, 0]: Acquisition function value of XY
        # [1, 1]: Second GP, mean and variance
        cm = self.ax[0, 0].contourf(*inputsg, mean, cmap='jet', levels=n_contours)  # Normalized across plots
        cv = self.ax[0, 1].contourf(*inputsg, variance, levels=n_contours)
        self.ax[0, 1].scatter(points[:, 0], points[:, 1])
        ca = self.ax[1, 0].contourf(*inputsg, ucb, levels=n_contours)
        cg = self.ax[1, 1].plot(samples1D, mean2)
        self.ax[1, 1].fill_between(samples1D, mean2+variance2, mean2-variance2, facecolor='blue', alpha=0.5)
        

        # formatting
        self.ax[0, 0].set_aspect('equal')
        self.ax[0, 0].set_title('Mean')
        self.ax[0, 0].set_ylabel('$y~[m]$')
        self.ax[0, 0].set_xlabel('$x~[m]$')
                
        self.ax[0, 1].set_aspect('equal')
        self.ax[0, 1].set_title('Variance')
        self.ax[0, 1].set_ylabel('$y~[m]$')
        self.ax[0, 1].set_xlabel('$x~[m]$')
                
        self.ax[1, 0].set_aspect('equal')
        self.ax[1, 0].set_title('Reward XY')
        self.ax[1, 0].set_ylabel('$y~[m]$')
        self.ax[1, 0].set_xlabel('$x~[m]$')
        
        self.ax[1, 1].set_title('Heading GP')
        self.ax[1, 1].set_ylabel('$Value~$')
        self.ax[1, 1].set_xlabel('$\theta~[rad]$')
        
        self.ax[0, 0].set_xlim(self.bounds[0], self.bounds[1])
        self.ax[0, 0].set_ylim(self.bounds[3], self.bounds[2])
        self.ax[0, 0].grid(True)
        
        self.ax[0, 1].set_xlim(self.bounds[0], self.bounds[1])
        self.ax[0, 1].set_ylim(self.bounds[3], self.bounds[2])
        self.ax[0, 1].grid(True)
        
        self.ax[1, 0].set_xlim(self.bounds[0], self.bounds[1])
        self.ax[1, 0].set_ylim(self.bounds[3], self.bounds[2])
        self.ax[1, 0].grid(True)
        
        self.ax[1, 1].set_xlim(-np.pi, np.pi)
        self.ax[1, 1].grid(True)
        
        # Send out collection object, in iterable with zorder (as accepted
        # by FuncAnimation)
        graph_list = []
        graph_list.append(cm.collections)
        graph_list.append(cv.collections)
        graph_list.append(ca.collections)
        graph_list.append(cg)
        graph_list = [item for sublist in graph_list for item in sublist]
        
        # time taken
        print(time.time() - t1) 

        return graph_list
    

def move_figure(f, x, y):
    """ moves a plotted figure regardless of backend used

    Args:
        f (matplotlib.pyplot.figure): figure
        x (int): pixel position
        y (int): pixel position
    """
    backend = matplotlib.get_backend()
    if backend == 'TkAgg':
        f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
    elif backend == 'WXAgg':
        f.canvas.manager.window.SetPosition((x, y))
    else:
        # This works for QT and GTK
        # You can also use window.setGeometry
        f.canvas.manager.window.move(x, y)
        

if __name__ == "__main__":
    
    rospy.init_node("Visualization_node")
    try:
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(9,7))
        move_figure(fig, 1800, 0)
        ud = UpdateDist(ax)
        anim = FuncAnimation(fig, ud, interval=3000, blit=False)
        plt.tight_layout()
        plt.show()
    
    except:
        print("Plotting not working as intended.")
