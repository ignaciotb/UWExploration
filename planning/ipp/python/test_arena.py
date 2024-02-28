import dubins
import numpy as np
import matplotlib.pyplot as plt
import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import std_msgs.msg

import torch

"""
### TESTING DUBINS PATH GENERATION

p1 = [0, 0, 0]
p2 = [10, 0, 0]

path = dubins.shortest_path(p1, p2, 0.1)

wp, l = path.sample_many(0.5)

print(l[-1])
#print(wp)

arr2 = np.array(wp)
x = arr2[:, 0]
y = arr2[:, 1]
#print(x)
#print(y)

plt.plot(x, y)
plt.show()

#print(dubins.dubins_path_length(path))

### TESTING SAMPLING DIMENSIONS


bounds = [0, 10, 5, -5]

n_samples = 3

samples = np.random.uniform(low=[bounds[0], bounds[3]], high=[bounds[1], bounds[2]], size=[n_samples, 2])

print(samples)
h = std_msgs.msg.Header()
#h.stamp = rospy.Time.now()
lm_path = Path()
lm_path.header = h

for sample in samples:
    wp = PoseStamped()
    wp.header = h
    wp.pose.position.x = sample[0]
    wp.pose.position.y = sample[1]
    lm_path.append()
    
"""

bounds = [0, 10, 5, -5]
b = torch.tensor([[bounds[0], bounds[3]], [bounds[1], bounds[2]]])
print(b)