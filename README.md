# AUV Exploration

Collection of ROS packages for localization, map building and SLAM with autonomous underwater vehicles and sonar sensing.

## Dependencies (tested on Ubuntu 16.04 and 18.04)
* AUVLIB [here](https://github.com/ignaciotb/auvlib/tree/extended_bm) And set the same cmake flags required for Ubuntu 18.
* Bathymetric SLAM [here](https://github.com/ignaciotb/bathymetric_slam)
* GTSAM [here](https://github.com/borglab/gtsam)

```
sudo apt install python3-pygame python3-scipy python3-configargparse python3-numpy
pip install configargparse pygame 
```

If you're going to be working with Gaussian Processes maps, also install
* Pytorch [here](https://pytorch.org/)
* TensorFlow2 [here](https://www.tensorflow.org/install)
* GPflow latest source from Github [here](https://github.com/GPflow/GPflow)
```
pip install gpytorch open3d 
```
If you want to try waypoint navigation for an AUV, clone this repo within your catkin workspace to plan missions in RVIZ
* Waypoint_navigation_plugin [here](https://github.com/KumarRobotics/waypoint_navigation_plugin)

## Building
This is a collection of ROS packages. Just clone the repo within your catking workspace and run
```
rosdep install --from-paths catkin_ws --ignore-src --rosdistro=$ROS_DISTRO -y
catkin_make -DCMAKE_BUILD_TYPE=Release install
```
