#ifndef BATHYMAP_CONSTRUCTOR_HPP
#define BATHYMAP_CONSTRUCTOR_HPP

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <cereal/archives/binary.hpp>

#include "data_tools/std_data.h"

#include "submaps_tools/cxxopts.hpp"
#include "submaps_tools/submaps.hpp"
#include "registration/utils_visualization.hpp"

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/TransformStamped.h>

using namespace Eigen;
using namespace std;

class BathymapConstructor{

public:
    BathymapConstructor(std::string node_name, ros::NodeHandle &nh);
    ~BathymapConstructor();

    void init(const boost::filesystem::path map_path, const boost::filesystem::path auv_path);

    void run();

    void broadcastTf(const ros::TimerEvent &event);

private:
    std::string node_name_;
    ros::NodeHandle* nh_;

    ros::Publisher ping_pub_;
    ros::Publisher map_pub_;
    ros::Publisher test_pub_;
//    ros::Subscriber mbes_laser_sub_;

    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener* tfListener_;
    tf::TransformListener tflistener_;

    Eigen::Isometry3f map_tf_;
    Eigen::Isometry3f odom_tf_;

    SubmapsVec maps_gt_;
    SubmapsVec traj_pings_;

    std::vector<geometry_msgs::TransformStamped> pings_tf_;

    int ping_num_;
};


#endif
