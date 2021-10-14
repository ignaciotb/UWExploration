#ifndef BATHY_SLAM_HPP
#define BATHY_SLAM_HPP

#include <ros/ros.h>
#include <iostream>

#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/io/ply_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>
#include <std_srvs/Empty.h>

#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf/transform_broadcaster.h>

#include <bathy_graph_slam/LandmarksIdx.h>

#include "submaps_tools/submaps.hpp"
#include "data_tools/std_data.h"
#include "data_tools/benchmark.h"

#include <actionlib/client/simple_action_client.h>
#include <Eigen/Core>

#include <bathy_graph_slam/sam_graph.hpp>
#include <bathy_graph_slam/serial.hpp> 

class BathySlamNode
{
    typedef std::tuple<sensor_msgs::PointCloud2Ptr, tf::Transform> ping_raw;

public:

    BathySlamNode(std::string node_name, ros::NodeHandle &nh);

    void odomCB(const nav_msgs::OdometryPtr &odom_msg);

    bool emptySrv(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res);

    void updateGraphCB(const sensor_msgs::PointCloud2Ptr &lm_pcl_msg, 
                       const bathy_graph_slam::LandmarksIdxPtr &lm_idx);

    Pose3 odomStep(double t_step, Pose3& current_pose);

    void checkForLoopClosures(SubmapObj submap_i);

    std::string node_name_;
    ros::NodeHandle *nh_;
    // ros::Publisher submaps_pub_;
    ros::Subscriber odom_subs_, submap_subs_;
    std::string map_frame_, odom_frame_, base_frame_, mbes_frame_, graph_init_path_, graph_solved_path_;
    ros::ServiceServer synch_service_;

    message_filters::Subscriber<sensor_msgs::PointCloud2> lm_subs_;
    message_filters::Subscriber<bathy_graph_slam::LandmarksIdx> lm_idx_subs_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, bathy_graph_slam::LandmarksIdx> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> *synch_;

    SubmapsVec submaps_vec_;
    std::vector<nav_msgs::Odometry> odomVec_;
    boost::shared_ptr<nav_msgs::Odometry> prev_submap_odom_;
    int submaps_cnt_;
    bool first_msg_;

    std::vector<tf::Transform> tf_submaps_vec_;
    tf::TransformBroadcaster submaps_bc_;
    tf::TransformListener tflistener_;
    tf::StampedTransform tf_map_odom_;
    tf2_ros::StaticTransformBroadcaster static_broadcaster_;
    // tf2_ros::Buffer tfBuffer_;
    // tf2_ros::TransformListener tfListener_;

    boost::shared_ptr<samGraph> graph_solver;
};

#endif
