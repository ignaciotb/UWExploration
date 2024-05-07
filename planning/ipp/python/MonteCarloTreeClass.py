import rospy
import numpy as np

class Node(object):
    
    def __init__(self, pose, depth, parent = None) -> None:
        self.pose           = pose
        self.depth          = depth
        self.parent         = parent
        self.children       = []
        self.reward         = 0
        self.visit_count    = 0
    
    
class MonteCarloTree(object):
    
    def __init__(self, start_pose, gp, border_margins, horizon_distance, bounds) -> None:
        self.root = Node(start_pose)
        self.gp = gp
        self.horizon_distance = horizon_distance
        self.border_margins = border_margins
        self.global_bounds = bounds
        self.C = 2.0
        self.max_depth = 3
        
    def iterate(self):
            
        # run iteration
        # 1. select a node
        # 2. expand node
        # 3. simulate for children
        # 4. backpropagate
        pass
    
    def select_node(self):
        # At each iteration, select a node to expand
        node = self.root
        while node.children != [] and node.depth < self.max_depth:
            children_uct = np.zeros(len(node.children))
            for id, child in enumerate(node.children):
                children_uct[id] = self.UCT(child)
            max_id = np.argmax(children_uct)
            node = node.children[max_id]
        return node
                
        
    def UCT(self, node):
        if node.visit_count == 0:
            return np.inf
        return node.reward / node.visit_count + self.C * np.sqrt(np.log(self.root.visit_count)/node.visit_count)
    
    def expand_node(self, node):
        # Get children of node through BO with multiple candidates
        self.XY_acqf                = UCB_xy(model=gp_terrain, beta=self.beta)
        self.bounds_XY_torch        = torch.tensor([[bounds[0], bounds[3]], [bounds[1], bounds[2]]]).to(torch.float)
        candidates, _ = optimize_acqf(acq_function=self.XY_acqf, bounds=self.bounds_XY_torch, q=1, num_restarts=max_iter, raw_samples=nbr_samples)
        
    
    def simulate_node(self, node):
        # Randomly sample around node to get expected value
        pass
    
    def backpropagate(self, node):
        # Push reward updates through tree upwards after simulating
        pass
    

root = Node([0, 0, np.pi], 0)
tree = MonteCarloTree(root)
        