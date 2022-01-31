# AUV Exploration

Collection of ROS packages for localization, map building and SLAM with autonomous underwater vehicles and sonar sensing.

## Dependencies (tested on Ubuntu 20.04)
* ROS Noetic
* AUVLIB [here](https://github.com/nilsbore/auvlib) And set the same cmake flags required for Ubuntu 18.
* Bathymetric SLAM [here](https://github.com/ignaciotb/bathymetric_slam)
* GTSAM [here](https://github.com/borglab/gtsam)

```
sudo apt install python3-pygame python3-scipy python3-configargparse python3-numpy
pip install configargparse pygame 
```
Make sure your scipy version is >= 1.4.0

If you're going to be working with Gaussian Processes maps, also install
* Pytorch [here](https://pytorch.org/)
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
## Troubleshooting
If you experience errors with GTSAM libraries not being found, add this line at the end of your .bashrc

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

## Running 
### Basic demo with one AUV
Run the AUV model with an MBES in a given underwater scenario
```
roslaunch auv_model auv_environment.launch
roslaunch auv_model auv_env_aux.launch
```
You should see in RVIZ the AUV and it's MBES pings.

### Manual vs waypoint navigation with one AUV
In order to run the AUV manually with the keyboard (w=forward, s=backward a,d=+/-yaw, up,down=+/-pitch)
```
roslaunch basic_navigation basic_mission.launch manual_control:=True
```
Alternatively, to plan and execute autonomous waypoint navigation missions, install [this package](https://github.com/KumarRobotics/waypoint_navigation_plugin).
```
roslaunch basic_navigation basic_mission.launch manual_control:=False
```
And add and publish waypoints through RVIZ as in their tutorial.

### Manual navigation with multiple AUVs
Example of multiagent mission
```
roslaunch auv_model auv_environment.launch namespace:=hugin_0
roslaunch auv_model auv_environment.launch namespace:=hugin_1 y:=10
roslaunch auv_model auv_environment.launch namespace:=hugin_2 y:=-10
roslaunch auv_model auv_env_aux.launch
roslaunch basic_navigation basic_mission.launch manual_control:=True
```
*WP navigation isn't implemented yet for several AUVs.

### Particle filter localization with two AUVs
Check 'auv_pf.launch' for the main filter parameters
```
roslaunch auv_particle_filter auv_pf.launch namespace:=hugin_0 x:=-300 y:=-400
roslaunch auv_particle_filter auv_pf.launch namespace:=hugin_1 x:=-330 y:=-430
roslaunch auv_model auv_env_aux.launch
roslaunch basic_navigation basic_mission.launch manual_control:=True
```
### RBPF SLAM
Check 'auv_pf.launch' for the main filter parameters. A decent GPU is required for this one.
```
roslaunch rbpf_slam rbpf_slam.launch particle_count:=5
roslaunch auv_model auv_env_aux.launch
roslaunch basic_navigation basic_mission.launch manual_control:=False
```
### Submap graph SLAM
Coming
