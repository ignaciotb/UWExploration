import numpy as np
from botorch.acquisition import qSimpleRegret, UpperConfidenceBound, qUpperConfidenceBound, qLowerBoundMaxValueEntropy, qKnowledgeGradient, qNoisyExpectedImprovement
import torch
from botorch.optim import optimize_acqf
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import rospy
import ipp_utils
import GaussianProcessClass
import pickle

""" PLAN FOR IMPLEMENTING MULTIPLE TRAINING GPs

1. When the node object is created, we simulate MBES points and store those
    - generate_points
    
2. We create an action server for the node, which can give these simulated MBES points
    - action_cb
    
3. In the frozen_gp, create a training method that can sample from both the
simulated as and the real action servers. Split 50/50, remove the gp points when they've been used.

4. Set a while loop that calls the training of the GP until points are exhausted. Until it has finished training,
do not let the node expand to new children, but instead keep doing rollouts from it (which will iteratively get better).

"""


class Node(object):
    
    def __init__(self, position, depth, id_nbr = 0, parent = None, gp = None) -> None:
        self.position           = position
        self.depth              = depth
        self.parent             = parent
        self.children           = []
        self.reward             = -np.inf
        self.visit_count        = 0
        self.id                 = (depth ** 2) * id_nbr
        self.training           = False
        self.device             = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gp                 = gp
        #self.gp.model.to(self.device)
        self.map_frame          = rospy.get_param("~map_frame")
        self.simulated_points   = np.empty((0,3))
        
        # If not root, we generate and train on simulated points
        if id_nbr > 0:
            points = ipp_utils.generate_points(self.gp, self.parent.position, self.position)
            self.gp.simulated_beams = np.concatenate((points, self.gp.simulated_beams), axis=0)
        training_iteration = 0      
        while training_iteration < 40:
            self.gp.train_simulated_and_real_iteration()
            training_iteration += 1
    
    
    
class MonteCarloTree(object):
    
    def __init__(self, start_position, gp, beta, border_margin, horizon_distance, bounds) -> None:
        self.root                       = Node(position=start_position, depth=0, gp=gp)
        self.beta                       = beta
        self.horizon_distance           = horizon_distance
        self.border_margin              = border_margin
        self.global_bounds              = bounds
        self.iteration                  = 0
        self.C                          = rospy.get_param("~MCTS_UCT_C")
        self.max_depth                  = 1 #rospy.get_param("~MCTS_max_depth")
        self.rollout_reward_distance    = rospy.get_param("~swath_width")/2.0
                
    def iterate(self):
        # run iteration
        # 1. select a node
        node = self.select_node()
        # If node has not been visited, rollout to get reward
        if node.visit_count == 0 and node != self.root:
            value = self.rollout_node(node)
            self.backpropagate(node, value)
            
        # If node has been visited and there is reward, then expand children
        else:
            # If max depth not hit, and not still training, expand
            if node.depth < self.max_depth:
                #print("Called expand node for node: " + str(node.id))
                self.expand_node(node)
                for child in node.children:
                    value = self.rollout_node(child)
                    self.backpropagate(child, value)
        # backpropagate
        self.iteration += 1
    
    def get_best_solution(self):
        values = []
        for child in self.root.children:
            values.append(child.reward)
        max_ind = np.argmax(values)
        max_int = len(values)
        return self.root.children[np.random.randint(0, max_int)]
    
    def select_node(self):
        # At each iteration, select a node to expand
        node = self.root
        while node.children != [] and node.depth <= self.max_depth:
            children_uct = np.zeros(len(node.children))
            for id, child in enumerate(node.children):
                children_uct[id] = self.UCT(child)
            max_id = np.argmax(children_uct)
            node = node.children[max_id]
        return node
                
        
    def UCT(self, node):
        # Calculate tree based UCB
        if node.visit_count == 0:
            return np.inf
        return node.reward / node.visit_count + self.C * np.sqrt(np.log(self.root.visit_count)/node.visit_count)
    
    def expand_node(self, node, nbr_children = 5):
        # Get children of node through BO with multiple candidates
        
        # Use tree GP to set up acquisition function (needs to be MC enabled, to return q candidates)
        # Use node position to set up dynamic bounds
        # Optimize the acqfun, return several candidates
        
        #print("expanding, parent at node depth: " + str(node.depth))

        #node.gp.model.model.to(node.device)
        #print(node.gp.model.model.variational_strategy.variational_distribution.covariance_matrix.device)
        #print(node.gp.model.model.variational_strategy.variational_distribution.loc.device)
        #print(node.gp.model.model.covar_module)

        #print(next(node.gp.model.parameters()).device)

        #node.gp.model = node.gp.model.to(node.device)
        #node.gp.model.model = node.gp.model.model.to(node.device)
        #node.gp.model.model.mean_module.mean_prior.loc = node.gp.model.model.mean_module.mean_prior.loc.to(node.device)
        #node.gp.model.model.variational_strategy = node.gp.model.model.variational_strategy.to(node.device)
        #node.gp.model.model.variational_strategy.variational_distribution.covariance_matrix = node.gp.model.model.variational_strategy.variational_distribution.covariance_matrix.to(node.device)
        #print(next(node.gp.model.parameters()).device)
        #print(node.gp.model.model.covar_module)
        #print(node.gp.model.model.variational_strategy.variational_distribution)
        #print(node.gp.model.model.variational_strategy.inducing_points.device)
        #print(node.gp.model.model.variational_strategy.variational_distribution.loc.device)
        #print(node.gp.model.model.variational_strategy.variational_distribution.covariance_matrix.device)
        #print(node.gp.model.model.variational_strategy..device)

        

        #node.gp.model.to(node.device).float()
        #node.gp.likelihood.to(node.device)
        #XY_acqf         = qSimpleRegret(model=node.gp.model)                                # Just pure exploration (but also gets stuck on maxima)
        XY_acqf         = qUpperConfidenceBound(model=node.gp.model, beta=self.beta)       # Issues with getting stuck in local maxima
        #XY_acqf         = qKnowledgeGradient(model=node.gp.model)                          # No, cant fantasize with variational GP

        #XY_acqf.to(node.device)
        
        #node.gp.model.model.variational_strategy.inducing_points.data = node.gp.inducing_pts_copy.to(node.device).float()

        #print(node.gp.model.model.covar_module.
        
        local_bounds    = ipp_utils.generate_local_bounds(self.global_bounds, node.position, self.horizon_distance, self.border_margin)
        
        bounds_XY_torch = (torch.tensor([[local_bounds[0], local_bounds[1]], [local_bounds[2], local_bounds[3]]]).to(torch.float)).to(node.device)


        #XY_acqf.model.to(node.device)
        #bounds_XY_torch.to(node.device)
        #print(node.device)
        #print(bounds_XY_torch.is_cuda)
        


        t1 = time.time()




        candidates, _   = optimize_acqf(acq_function=XY_acqf, bounds=bounds_XY_torch, q=nbr_children, num_restarts=5, raw_samples=100)
        
        print("****** TIME TAKEN TO OPTIMIZE ALL CANDIDATES: " + str(time.time() - t1) + " ******")
        ipp_utils.save_model(node.gp.model, "Parent_gp.pickle")
        t2 = time.time()
        
        for i in range(nbr_children):
            new_gp = GaussianProcessClass.frozen_SVGP()
            #new_gp.model.to(node.device)
            cp = torch.load("Parent_gp.pickle", map_location=node.device)
            new_gp.model.load_state_dict(cp['model'])
            new_gp.model.to(node.device)
            #new_gp.model = ipp_utils.load_model(new_gp.model, "Parent_gp.pickle", device=node.device)
            new_gp.real_beams = node.gp.real_beams
            new_gp.simulated_beams = node.gp.simulated_beams
            #new_gp.inducing_pts_copy = node.gp.inducing_pts_copy
            n = Node(position=list(candidates[i,:].cpu().detach().numpy()), id_nbr=i+1,depth=node.depth + 1, parent=node, gp=new_gp)
            node.children.append(n)
        print("****** TIME TAKEN TO EXPAND ALL NODES: " + str(time.time() - t2) + " ******")
        
    
    def rollout_node(self, node):
        
        
        # Randomly sample from node to terminal state to get expected value
        # we dont want to have to go to terminal state, instead 
        
        local_bounds = ipp_utils.generate_local_bounds(self.global_bounds, node.position, self.rollout_reward_distance, self.border_margin)
        samples_np = np.random.uniform(low=[local_bounds[0], local_bounds[1]], high=[local_bounds[2], local_bounds[3]], size=[20, 2])
        samples_torch = (torch.from_numpy(samples_np).type(torch.FloatTensor)).unsqueeze(-2)
        
        acq_fun = UpperConfidenceBound(model=node.gp.model, beta=self.beta)
        ucb = acq_fun.forward(samples_torch)
        reward = ucb.mean().item() - node.depth
        return reward
    
    def backpropagate(self, node, value):
        # Push reward updates through tree upwards after simulating
        current = node
        reward = max(current.reward, value)
        current.reward = reward
        current.visit_count += 1
        while current.depth > 0:
            current = current.parent
            if reward > current.reward:
                current.reward = reward
            current.visit_count += 1
            
    def show_tree(self):
        nodes = [self.root]
        x = " "
        norm = mpl.colors.Normalize(vmin=0, vmax=self.max_depth)
        cmap = cm.Wistia
        m = cm.ScalarMappable(norm=norm, cmap=cmap)
        while len(nodes) > 0:
            current = nodes.pop(0)
            print("N, d=" + str(current.depth) + " " + str(current.depth*4*x) + str(round(current.reward, 4)) + ", visited " +str(current.visit_count))
            plt.scatter(current.position[0], current.position[1], s=200-40*current.depth, c=np.array([m.to_rgba(current.depth)[:3]]))
            for i, child in enumerate(current.children):
                plt.plot([current.position[0], child.position[0]], [current.position[1], child.position[1]], linewidth=3-0.5*current.depth, color=m.to_rgba(child.depth))
                nodes.insert(i,child)
        current = self.root
        while current.depth < self.max_depth:
            max_id = 0
            max_val = -np.inf
            for i, child in enumerate(current.children):
                if child.reward > max_val:
                    max_id = i
                    max_val = child.reward
            if len(current.children) > 0:
                plt.plot([current.position[0], current.children[max_id].position[0]], [current.position[1], current.children[max_id].position[1]], linewidth=3, color="red")
            print(current.depth)
            print(self.max_depth)
            current = current.children[max_id]
        
        # Plotting params
        n = 50
        n_contours = 25

        # posterior sampling locations for first GP
        inputsg = [
            np.linspace(self.global_bounds[0], self.global_bounds[1], n),
            np.linspace(self.global_bounds[3], self.global_bounds[2], n)
        ]
        inputst = np.meshgrid(*inputsg)
        s = inputst[0].shape
        inputst = [_.flatten() for _ in inputst]
        inputst = np.vstack(inputst).transpose()
        
        ucb_fun = UCB_xy(model1, beta=self.beta)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gp.to(device).float()
        torch.cuda.empty_cache()
        
        
        ucb_list = []
        divs = 10
        with torch.no_grad():
            for i in range(0, divs):
                # sample
                inputst_temp = torch.from_numpy(inputst[i*int(n*n/divs):(i+1)*int(n*n/divs), :]).to(device).float()
                mean_r, sigma_r = ucb_fun._mean_and_sigma(inputst_temp)
                ucb = (abs(mean_r - self.gp.model.mean_module.constant)) + self.beta * sigma_r
                ucb_list.append(ucb.cpu().numpy())

        ucb = np.vstack(ucb_list).reshape(s)
        
        ax = plt.gca()
        ax.set_aspect('equal')
        fig = plt.gcf()
        ca = ax.contourf(*inputsg, ucb, levels=n_contours)
        fig.colorbar(ca, ax=ax)
        plt.show()
            

if __name__== "__main__":
    model1 = ipp_utils.load_model("/home/alex/.ros/GP_env.pickle")

    bounds = [592, 821, -179, -457]

    tree = MonteCarloTree(start_position=[639, -204], gp=model1, beta=5.5, 
                          border_margin=30, horizon_distance=100, bounds=bounds)

    t1 = time.time()
    while time.time() - t1 < 10:
        node = tree.iterate()
    t2 = time.time()
    print(t2-t1)
    tree.show_tree()

            
