import numpy as np
from botorch.acquisition import qSimpleRegret, UpperConfidenceBound
import torch
from botorch.optim import optimize_acqf
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm
import rospy
import ipp_utils

class Node(object):
    
    def __init__(self, position, depth, parent = None, gp = None) -> None:
        self.position       = position
        self.depth          = depth
        self.parent         = parent
        self.gp             = gp
        self.children       = []
        self.reward         = -np.inf
        self.visit_count    = 0
    
    def generate_points(self):
        pass
    
    def train_separate_points(self):
        pass
    
    
class MonteCarloTree(object):
    
    def __init__(self, start_position, gp, beta, border_margin, horizon_distance, bounds) -> None:
        self.root                       = Node(position=start_position, depth=0)
        self.gp                         = gp
        self.beta                       = beta
        self.horizon_distance           = horizon_distance
        self.border_margin              = border_margin
        self.global_bounds              = bounds
        self.iteration                  = 0
        self.C                          = rospy.get_param("~MCTS_UCT_C")
        self.max_depth                  = rospy.get_param("~MCTS_max_depth")
        self.MCTS_sample_decay_factor   = rospy.get_param("~MCTS_sample_decay_factor")
        self.rollout_reward_distance    = rospy.get_param("~swath_width")/2.0
                
    def iterate(self):
        # run iteration
        # 1. select a node
        node = self.select_node()
        # If node has not been visited, rollout to get reward
        if node.visit_count == 0 and node != self.root:
            value = self.rollout_node(node)
        # If node has been visited and there is reward, then expand children
        else:
            # If max depth not hit, expand
            if node.depth < self.max_depth:
                self.expand_node(node)
                node = node.children[0]
                
            value = self.rollout_node(node)
        # backpropagate
        self.backpropagate(node, value)
        self.iteration += 1
    
    def get_best_solution(self):
        values = []
        for child in self.root.children:
            values.append(child.reward)
        max_ind = np.argmax(values)
        return self.root.children[max_ind]
    
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
    
    def expand_node(self, node, nbr_children = 3):
        # Get children of node through BO with multiple candidates
        
        # Use tree GP to set up acquisition function (needs to be MC enabled, to return q candidates)
        # Use node position to set up dynamic bounds
        # Optimize the acqfun, return several candidates
        
        #print("expanding, parent at node depth: " + str(node.depth))
        
        #XY_acqf         = qSimpleRegret(model=self.gp)
        XY_acqf         = UpperConfidenceBound(model=self.gp, beta=self.beta)
        
        local_bounds    = ipp_utils.generate_local_bounds(self.global_bounds, node.position, self.horizon_distance, self.border_margin)
        
        bounds_XY_torch = torch.tensor([[local_bounds[0], local_bounds[1]], [local_bounds[2], local_bounds[3]]]).to(torch.float)
        print(bounds_XY_torch)
        
        decayed_samples = max(10, 50-self.iteration*self.MCTS_sample_decay_factor)
        
        candidates, _   = optimize_acqf(acq_function=XY_acqf, bounds=bounds_XY_torch, q=nbr_children, num_restarts=4, raw_samples=decayed_samples)
        
        
        for i in range(nbr_children):
            n = Node(list(candidates[i,:].cpu().detach().numpy()), node.depth + 1, node)
            node.children.append(n)
    
    def rollout_node(self, node):
        
        #print("rolling out, for node at depth: " + str(node.depth))
        
        # Randomly sample from node to terminal state to get expected value
        # we dont want to have to go to terminal state, instead 
        
        local_bounds = ipp_utils.generate_local_bounds(self.global_bounds, node.position, self.rollout_reward_distance, self.border_margin)
        
        # Adjust for if the area is smaller due to bounds, this should be penalized
        samples_np = np.random.uniform(low=[local_bounds[0], local_bounds[1]], high=[local_bounds[2], local_bounds[3]], size=[20, 2])
        samples_torch = (torch.from_numpy(samples_np).type(torch.FloatTensor)).unsqueeze(-2)
        
        acq_fun = UpperConfidenceBound(model=node.gp, beta=self.beta)
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
    model1 = pickle.load(open(r"/home/alex/.ros/GP_env.pickle","rb"))

    bounds = [592, 821, -179, -457]

    tree = MonteCarloTree(start_position=[639, -204], gp=model1, beta=5.5, 
                          border_margin=30, horizon_distance=100, bounds=bounds)

    t1 = time.time()
    while time.time() - t1 < 10:
        node = tree.iterate()
    t2 = time.time()
    print(t2-t1)
    tree.show_tree()

            
