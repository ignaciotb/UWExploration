import pickle
from gpytorch.likelihoods import GaussianLikelihood
from matplotlib import pyplot as plt
from botorch.acquisition import UpperConfidenceBound
#from AcquisitionFunctionClass import UCB_path
import torch
import numpy as np

# Load MBES points
#MBES = pickle.load(open(r"/home/alex/.ros/Mon, 22 Apr 2024 15:24:15_iteration_2312_MBES.pickle", "rb"))
#print(MBES.shape)

# Load first model
model1 = pickle.load(open(r"/home/alex/.ros/GP_env_vis.pickle","rb"))
likelihood1 = GaussianLikelihood()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
likelihood1.to(device).float()
model1.to(device).float()
ind_points = model1.model.variational_strategy.inducing_points.detach().numpy()
torch.cuda.empty_cache()

# Load second model
"""
model2 = pickle.load(open(r"/home/alex/.ros/Fri, 19 Apr 2024 10:54:09_iteration_2889_GP_path.pickle","rb"))
model2.eval()
model2.likelihood.eval()
likelihood2 = GaussianLikelihood()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
likelihood2.to(device).float()
model2.to(device).float()
"""
# Plotting params
n = 200
n_contours = 50

# posterior sampling locations for first GP
inputsg = [
    np.linspace(min(ind_points[:, 0])-15, max(ind_points[:, 0])+15, n),
    np.linspace(min(ind_points[:, 1])-15, max(ind_points[:, 1])+15, n)
]
inputst = np.meshgrid(*inputsg)
s = inputst[0].shape
inputst = [_.flatten() for _ in inputst]
inputst = np.vstack(inputst).transpose()

# posterior sampling locations for second GP
inputsg2 = [
    np.linspace(-155 - 40, -155 + 40, n),
    np.linspace(-47 - 40, -47 + 40, n),
    np.linspace(0, 2*np.pi, 8)
]
inputst2 = np.meshgrid(*inputsg2)
s2 = inputst2[0].shape
inputst2 = [_.flatten() for _ in inputst2]
inputst2 = np.vstack(inputst2).transpose()

# Define acquisition functions
ucb_fun = UpperConfidenceBound(model1, 20)
"""
ucb_path = UpperConfidenceBound(model2, 10)
"""
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

"""
# Outputs for GP 2
ucb2_list = []
divs = 10
with torch.no_grad():
    for i in range(0, divs):
        # sample
        inputst_temp2 = torch.from_numpy(inputst2[i*int(n*n*8/divs):(i+1)*int(n*n*8/divs), :]).to(device).float()
        mean_r, sigma_r = ucb_path._mean_and_sigma(inputst_temp2)
        ucb2 = mean_r + ucb_path.beta.sqrt() * sigma_r
        ucb2_list.append(ucb2)
        print(i)
        
        
ucb2 = np.vstack(ucb2_list).reshape(s2)

print("Donezo")
"""

# First plots
# plot raw, mean, and variance
#levels = np.linspace(min(targets), max(targets), n_contours)
fig, ax = plt.subplots(sharex=True, sharey=True)
#cr = ax[0].scatter(MBES[:, 0], MBES[:, 1], c = MBES[:, 2],
#                    cmap='jet', s=0.4, edgecolors='none')
#cm = ax.contourf(*inputsg, mean, cmap='jet', levels=n_contours)  # Normalized across plots
cm = ax.contourf(*inputsg, mean, cmap='jet', levels=n_contours)
#cv = ax.contourf(*inputsg, variance, levels=n_contours)
indpts = model1.model.variational_strategy.inducing_points.data.cpu().numpy()
ax.plot(indpts[:, 0], indpts[:, 1], 'ko', markersize=1, alpha=0.5)
#ca = ax.contourf(*inputsg, ucb, levels=n_contours)



#post_cloud = np.hstack((inputst, mean.reshape(-1, 1)))
#np.save("./posterior.npy", post_cloud)

# colorbars
#fig.colorbar(cr, ax=ax[0])
fig.colorbar(cm, ax=ax)
#fig.colorbar(cv, ax=ax)
#fig.colorbar(ca, ax=ax)

# # formatting
ax.set_aspect('equal')
ax.set_title('Variance')
ax.set_ylabel('$y~[m]$')
ax.set_xlabel('$x~[m]$')

#ax[1].set_aspect('equal')
#ax[1].set_title('Mean')
#ax[1].set_ylabel('$y~[m]$')
#ax[2].set_aspect('equal')
#ax[2].set_title('Variance')
#ax[2].set_ylabel('$y~[m]$')
#ax[3].set_aspect('equal')
#ax[3].set_title('UCB')
#ax[3].set_ylabel('$y~[m]$')
#ax[3].set_xlabel('$x~[m]$')
#plt.tight_layout()


"""
# Second plots
# plot raw, mean, and variance
#levels = np.linspace(min(targets), max(targets), n_contours)
fig2, ax2 = plt.subplots(ncols=2, nrows=4, sharex=True, sharey=True)
ucb2 = np.array(ucb2)
c1 = ax2[0, 0].contourf(*inputsg2[:2], ucb2[:,:,0], levels=n_contours)
c2 = ax2[1, 0].contourf(*inputsg2[:2], ucb2[:,:,1], levels=n_contours)
c3 = ax2[2, 0].contourf(*inputsg2[:2], ucb2[:,:,2], levels=n_contours)
c4 = ax2[3, 0].contourf(*inputsg2[:2], ucb2[:,:,3], levels=n_contours)
fig2.colorbar(c1, ax=ax2[0, 0])
fig2.colorbar(c2, ax=ax2[1, 0])
fig2.colorbar(c3, ax=ax2[2, 0])
fig2.colorbar(c4, ax=ax2[3, 0])
ax2[0, 0].set_aspect('equal')
ax2[0, 0].set_title('UCB2')
ax2[0, 0].set_ylabel('$y~[m]$')
ax2[1, 0].set_aspect('equal')
ax2[1, 0].set_title('UCB2')
ax2[1, 0].set_ylabel('$y~[m]$')
ax2[2, 0].set_aspect('equal')
ax2[2, 0].set_title('UCB2')
ax2[2, 0].set_ylabel('$y~[m]$')
ax2[3, 0].set_aspect('equal')
ax2[3, 0].set_title('UCB2')
ax2[3, 0].set_ylabel('$y~[m]$')
ax2[3, 0].set_xlabel('$x~[m]$')

c5 = ax2[0, 1].contourf(*inputsg2[:2], ucb2[:,:,4], levels=n_contours)
c6 = ax2[1, 1].contourf(*inputsg2[:2], ucb2[:,:,5], levels=n_contours)
c7 = ax2[2, 1].contourf(*inputsg2[:2], ucb2[:,:,6], levels=n_contours)
c8 = ax2[3, 1].contourf(*inputsg2[:2], ucb2[:,:,7], levels=n_contours)
fig2.colorbar(c5, ax=ax2[0, 1])
fig2.colorbar(c6, ax=ax2[1, 1])
fig2.colorbar(c7, ax=ax2[2, 1])
fig2.colorbar(c8, ax=ax2[3, 1])
ax2[0, 1].set_aspect('equal')
ax2[0, 1].set_title('UCB2')
ax2[0, 1].set_ylabel('$y~[m]$')
ax2[1, 1].set_aspect('equal')
ax2[1, 1].set_title('UCB2')
ax2[1, 1].set_ylabel('$y~[m]$')
ax2[2, 1].set_aspect('equal')
ax2[2, 1].set_title('UCB2')
ax2[2, 1].set_ylabel('$y~[m]$')
ax2[3, 1].set_aspect('equal')
ax2[3, 1].set_title('UCB2')
ax2[3, 1].set_ylabel('$y~[m]$')
ax2[3, 1].set_xlabel('$x~[m]$')
plt.tight_layout()
"""


plt.show()

# Plot particle trajectory
#ax[0].plot(track[:,0], track[:,1], "-r", linewidth=0.2)

# # save
#fig.savefig(fname, bbox_inches='tight', dpi=1000)

# Free up GPU mem
del inputst
torch.cuda.empty_cache()
