import numpy as np
from AcquisitionFunctionClass import qUCB_xy, UCB_xy
import torch
from botorch.optim import optimize_acqf
import pickle
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.cm as cm

class Node(object):
    
    def __init__(self, position, depth, parent = None) -> None:
        self.position       = position
        self.depth          = depth
        self.parent         = parent
        self.children       = []
        self.reward         = 0
        self.visit_count    = 0
    
    
class MonteCarloTree(object):
    
    def __init__(self, start_position, gp, beta, border_margin, horizon_distance, bounds, max_depth,
                 MCTS_UCT_C, MCTS_sample_decay_factor) -> None:
        self.root                       = Node(position=start_position, depth=0)
        self.gp                         = gp
        self.beta                       = beta
        self.horizon_distance           = horizon_distance
        self.border_margin              = border_margin
        self.global_bounds              = bounds
        self.C                          = MCTS_UCT_C
        self.max_depth                  = max_depth
        self.iteration                  = 0
        self.MCTS_sample_decay_factor   = MCTS_sample_decay_factor
        
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
        
        XY_acqf         = qUCB_xy(model=self.gp, beta=self.beta)
        
        low_x           = max(self.global_bounds[0] + self.border_margin, min(node.position[0] - self.horizon_distance, node.position[0] + self.horizon_distance))
        high_x          = min(self.global_bounds[1] - self.border_margin, max(node.position[0] - self.horizon_distance, node.position[0] + self.horizon_distance))
        low_y           = max(self.global_bounds[3] + self.border_margin, min(node.position[1] - self.horizon_distance, node.position[1] + self.horizon_distance))
        high_y          = min(self.global_bounds[2] - self.border_margin, max(node.position[1] - self.horizon_distance, node.position[1] + self.horizon_distance))
        local_bounds    = [low_x, high_x, high_y, low_y]
        
        bounds_XY_torch = torch.tensor([[local_bounds[0], local_bounds[3]], [local_bounds[1], local_bounds[2]]]).to(torch.float)
        
        decayed_samples = max(10, 50-self.iteration*self.MCTS_sample_decay_factor)
        
        candidates, _   = optimize_acqf(acq_function=XY_acqf, bounds=bounds_XY_torch, q=nbr_children, num_restarts=4, raw_samples=decayed_samples)
        
        
        for i in range(nbr_children):
            n = Node(list(candidates[i,:].cpu().detach().numpy()), node.depth + 1, node)
            node.children.append(n)
    
    def rollout_node(self, node):
        
        #print("rolling out, for node at depth: " + str(node.depth))
        
        # Randomly sample from node to terminal state to get expected value
        # we dont want to have to go to terminal state, instead 
        low_x   = max(self.global_bounds[0] + self.border_margin, min(node.position[0] - self.horizon_distance, node.position[0] + self.horizon_distance))
        high_x  = min(self.global_bounds[1] - self.border_margin, max(node.position[0] - self.horizon_distance, node.position[0] + self.horizon_distance))
        low_y   = max(self.global_bounds[3] + self.border_margin, min(node.position[1] - self.horizon_distance, node.position[1] + self.horizon_distance))
        high_y  = min(self.global_bounds[2] - self.border_margin, max(node.position[1] - self.horizon_distance, node.position[1] + self.horizon_distance))
        
        # Adjust for if the area is smaller due to bounds, this should be penalized
        samples_np = np.random.uniform(low=[low_x, low_y], high=[high_x, high_y], size=[40, 2])
        samples_torch = (torch.from_numpy(samples_np).type(torch.FloatTensor)).unsqueeze(-2)
        
        acq_fun = UCB_xy(model=self.gp, beta=self.beta)
        ucb = acq_fun.forward(samples_torch)
        reward = ucb.mean().item()
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
            print("N, d=" + str(current.depth) + str(current.depth*4*x) + str(current.reward))
            plt.scatter(current.position[0], current.position[1], s=200-40*current.depth, c=np.array([m.to_rgba(current.depth)[:3]]))
            for i, child in enumerate(current.children):
                plt.plot([current.position[0], child.position[0]], [current.position[1], child.position[1]], linewidth=3-0.5*current.depth, color=m.to_rgba(child.depth))
                nodes.insert(i,child)
        current = self.root
        while current.depth < self.max_depth:
            max_id = 0
            max_val = 0
            for i, child in enumerate(current.children):
                if child.reward > max_val:
                    max_id = i
                    max_val = child.reward
            plt.plot([current.position[0], current.children[max_id].position[0]], [current.position[1], current.children[max_id].position[1]], linewidth=3, color="red")
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
        fig = plt.gcf()
        ca = ax.contourf(*inputsg, ucb, levels=n_contours)
        fig.colorbar(ca, ax=ax)
        plt.show()
            

if __name__== "__main__":
    model1 = pickle.load(open(r"/home/alex/.ros/GP_env.pickle","rb"))

    bounds = [592, 821, -179, -457]

    tree = MonteCarloTree(start_position=[750, -250], gp=model1, beta=9, border_margin=30, horizon_distance=100, bounds=bounds, max_depth=3)

    t1 = time.time()
    node = tree.iterate()
    t2 = time.time()
    print(t2-t1)
    tree.show_tree()





    """
    t1 = time.time()
    s = tree.select_node()
    t2 = time.time()
    tree.expand_node(s)
    t3 = time.time()
    tree.rollout_node(s)
    t4 = time.time()
    print(t2-t1)
    print(t3-t2)
    print(t4-t3)
    """

            