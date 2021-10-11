#!/usr/bin/python3

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plot
import numpy as np
import sys
from optparse import OptionParser

def parse_graph(input):
    poses = []
    landmarks = []
    for line in input:
        if line[0] == 'Pose':
            poses.append(line)
        else:
            landmarks.append(line)

    poses = np.vstack(poses)
    landmarks = np.vstack(landmarks)

    poses = np.asarray(poses[:,1:3]).astype(np.float)
    landmarks = np.asarray(landmarks[:,1:3]).astype(np.float)

    return (poses, landmarks)


parser = OptionParser()
parser.add_option("--initial", dest="initial",
                  default="", help="The filename that contains the SLAM problem.")
parser.add_option("--result", dest="result",
                  default="", help="The filename that contains the SLAM result.")
parser.add_option("--img", dest="img",
                  default="", help="The png image with the background bathy.")

parser.add_option("--output_file", dest="outputFile",
                  default="", help="The output file.")

(options, args) = parser.parse_args()

# Read the original and optimized poses files.
initial = None
if options.initial != '':
  initial = np.genfromtxt(options.initial, dtype=str, usecols = (0, 2, 3))

result = None
if options.result != '':
  result = np.genfromtxt(options.result, dtype=str, usecols = (0, 2, 3))

if options.img != '':
  plot.cla()
  img = plot.imread(options.img)
  plot.imshow(img, extent=[-647, 1081,
                          -1190, 523])

# figure, (ax1, ax2) = plot.subplots(1, 2)

if initial is not None:
 poses, landmarks = parse_graph(initial)

 plot.plot(poses[:, 0], poses[:, 1], '*', alpha=0.5, color="green")
 plot.plot(poses[0, 0], poses[0, 1], '*', alpha=0.5, color="blue")
 plot.plot(poses[:, 0], poses[:, 1], '-', alpha=0.5, color="green")
#  plot.plot(landmarks[:, 0], landmarks[:, 1], '*', alpha=0.5, color="red")
 #  ax1.set_xlim([-250, 200])
 #  ax1.set_ylim([-200, 300])

if result is not None:
 poses, landmarks = parse_graph(result)

 plot.plot(poses[:, 0], poses[:, 1], '*', alpha=0.5, color="green")
 plot.plot(poses[0, 0], poses[0, 1], '*', alpha=0.5, color="blue")
 plot.plot(poses[:, 0], poses[:, 1], '-', alpha=0.5, color="blue")
#  ax2.plot(landmarks[:, 0], landmarks[:, 1], '*', alpha=0.5, color="red")
 #  ax2.set_xlim([-250, 200])
 #  ax2.set_ylim([-200, 300])


#  plot.axis('equal')
#  plot.title('Trajectories: GT (Green)')
plot.show()