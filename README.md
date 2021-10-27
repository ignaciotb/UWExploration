# AUV Exploration

Collection of ROS packages for localization, map building and SLAM with autonomous underwater vehicles and sonar sensing.

## Dependencies (tested on Ubuntu 16.04 and 18.04)
* AUVLIB [here](https://github.com/ignaciotb/auvlib) 
* Bathymetric SLAM [here](https://github.com/ignaciotb/bathymetric_slam)

```
sudo apt install python3-pygame python3-scipy python3-configargparse python3-numpy
pip install open3d configargparse pygame
```

## Building
This is a collection of ROS packages. Just clone the repo within your catking workspace and run
```
rosdep install --from-paths catkin_ws --ignore-src --rosdistro=$ROS_DISTRO -y
catkin_make -DCMAKE_BUILD_TYPE=Release install
```
