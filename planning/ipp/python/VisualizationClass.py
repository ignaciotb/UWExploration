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


class UpdateDist(object):
    def __init__(self, ax):
        
        self.lock_environment_gp = filelock.FileLock("GP_env.pickle.lock")
        self.lock_heading_gp     = filelock.FileLock("GP_angle.pickle.lock")
        
        self.bounds = [-260, -40, 100, -70]
        
        self.ax = ax

        # Set up plot parameters
        self.ax[0, 0].set_xlim(self.bounds[0], self.bounds[1])
        self.ax[0, 0].set_ylim(self.bounds[3], self.bounds[2])
        self.ax[0, 0].grid(True)
        
        self.ax[0, 1].set_xlim(self.bounds[0], self.bounds[1])
        self.ax[0, 1].set_ylim(self.bounds[3], self.bounds[2])
        self.ax[0, 1].grid(True)
        
        self.ax[1, 0].set_xlim(self.bounds[0], self.bounds[1])
        self.ax[1, 0].set_ylim(self.bounds[3], self.bounds[2])
        self.ax[1, 0].grid(True)
        
        self.ax[1, 1].set_xlim(self.bounds[0], self.bounds[1])
        self.ax[1, 1].set_ylim(self.bounds[3], self.bounds[2])
        self.ax[1, 1].grid(True)
        
    def __call__(self, i):
                
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
        
        ucb_fun = UpperConfidenceBound(model1, 10)

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
                ucb = mean_r + ucb_fun.beta.sqrt() * sigma_r
                outputs = likelihood1(outputs)
                mean_list.append(outputs.mean.cpu().numpy())
                var_list.append(outputs.variance.cpu().numpy())
                ucb_list.append(ucb.cpu().numpy())

        mean = np.vstack(mean_list).reshape(s)
        variance = np.vstack(var_list).reshape(s)
        ucb = np.vstack(ucb_list).reshape(s)

        cm = self.ax[0, 0].contourf(*inputsg, mean, cmap='jet', levels=n_contours)  # Normalized across plots
        cv = self.ax[0, 1].contourf(*inputsg, variance, levels=n_contours)
        ca = self.ax[1, 0].contourf(*inputsg, ucb, levels=n_contours)

        # # formatting
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
        
        graph_list = []
        graph_list.append(cm.collections)
        graph_list.append(cv.collections)
        graph_list.append(ca.collections)
        graph_list = [item for sublist in graph_list for item in sublist]
        
        print(time.time() - t1) 
        
        return graph_list
    

def move_figure(f, x, y):
    """Move figure's upper left corner to pixel (x, y)"""
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
        anim = FuncAnimation(fig, ud, interval=3000, blit=True)
        plt.tight_layout()
        #cm = anim.init_func()
        #fig.colorbar(cm)
        plt.show()
    
    except:
        print("Plotting not working as intended.")
