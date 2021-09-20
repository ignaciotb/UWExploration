#ifndef BATHY_SLAM_HPP
#define BATHY_SLAM_HPP

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_srvs/Empty.h>

#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf/transform_broadcaster.h>

#include "submaps_tools/submaps.hpp"

#include "data_tools/std_data.h"
#include "data_tools/benchmark.h"

#include <actionlib/client/simple_action_client.h>

#include <Eigen/Core>

#include <bathy_graph_slam/sam_graph.hpp>

class BathySlamNode{
    typedef std::tuple<sensor_msgs::PointCloud2Ptr, tf::Transform> ping_raw;

public:

    BathySlamNode(std::string node_name, ros::NodeHandle &nh);

    bool emptySrv(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res);

    void updateTf();

    void pingCB(const sensor_msgs::PointCloud2Ptr &mbes_ping, 
                const nav_msgs::OdometryPtr &odom_msg);

    void enableCB(const std_msgs::BoolPtr &enable_msg);

    void addSubmap();

    Pose2 odomStep(unsigned int odom_step);

    std::string node_name_;
    ros::NodeHandle *nh_;
    ros::Publisher submaps_pub_;
    ros::Subscriber enable_subs_;
    std::string map_frame_, odom_frame_, base_frame_, mbes_frame_;
    ros::ServiceServer synch_service_;

    message_filters::Subscriber<sensor_msgs::PointCloud2> mbes_subs_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_subs_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> *synch_;

    SubmapsVec submaps_vec_;
    std::vector<ping_raw> submap_raw_;
    unsigned int submaps_cnt_;
    bool first_msg_;

    std::vector<tf::Transform> tf_submaps_vec_;
    tf::TransformBroadcaster submaps_bc_;
    tf::TransformListener tflistener_;
    tf::StampedTransform tf_mbes_base_;

    samGraph *isam_obj;
};

#endif
