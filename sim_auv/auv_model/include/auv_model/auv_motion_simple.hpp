#ifndef AUV_MOTION_SIMPLE_HPP
#define AUV_MOTION_SIMPLE_HPP

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <time.h>

#include <ros/ros.h>
//#include <pcl_ros/point_cloud.h>
//#include <pcl_ros/transforms.h>
//#include <pcl_conversions/pcl_conversions.h>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_msgs/Float64.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>
#include <sensor_msgs/PointCloud2.h>

#include <actionlib/client/simple_action_client.h>
#include <auv_2_ros/MbesSimAction.h>

using namespace Eigen;
using namespace std;

class AUVMotionModel{

public:
    AUVMotionModel(std::string node_name, ros::NodeHandle &nh);
    ~AUVMotionModel();

    void init();

    void updateMotion(const ros::TimerEvent &);

    void updateMeas(const ros::TimerEvent &);

private:
    std::string node_name_;
    ros::NodeHandle* nh_;

    ros::Publisher sim_ping_pub_;
    ros::Publisher odom_pub_;
    ros::Subscriber throttle_sub_;
    ros::Subscriber incl_sub_;
    ros::Subscriber thruster_sub_;

    tf2_ros::Buffer tfBuffer_;
    tf2_ros::TransformListener* tfListener_;
    tf::TransformListener tflistener_;
    tf::StampedTransform tf_map_odom_;
    tf::StampedTransform tf_base_mbes_;
    tf2_ros::TransformBroadcaster br_;

    actionlib::SimpleActionClient<auv_2_ros::MbesSimAction>* ac_;

    Eigen::Isometry3d map_tf_;
    Eigen::Isometry3d odom_tf_;
    nav_msgs::Odometry prev_odom_;

    ros::Time time_now_, time_prev_;
    geometry_msgs::TransformStamped prev_base_link_;
    geometry_msgs::TransformStamped new_base_link_;
//    geometry_msgs::TransformStamped tfmsg_map_odom_;

    std::string world_frame_, map_frame_, odom_frame_, base_frame_, mbes_frame_;

    double latest_thrust_, latest_throttle_, latest_inclination_;
    void thrustCB(const std_msgs::Float64ConstPtr& thrust_msg);
    void throttleCB(const std_msgs::Float64ConstPtr& throttle_msg);
    void inclinationCB(const std_msgs::Float64ConstPtr& inclination_msg);

};


#endif
