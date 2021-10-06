#ifndef SUBMAP_CONSTRUCTOR_HPP
#define SUBMAP_CONSTRUCTOR_HPP

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>

#include <iostream>

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
#include <bathy_graph_slam/LandmarksIdx.h>

#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf/transform_broadcaster.h>

#include "submaps_tools/submaps.hpp"
#include "registration/gicp_reg.hpp"

#include "data_tools/std_data.h"
#include "data_tools/benchmark.h"

#include <actionlib/client/simple_action_client.h>
#include <Eigen/Core>

namespace pcl
{
    template <>
    struct SIFTKeypointFieldSelector<PointXYZ>
    {
        inline float
        operator()(const PointXYZ &p) const
        {
            return p.z;
        }
    };
}

bool checkSiftOverlap(const corners& submap_i_corners, const Vector3d& sift_i){

    // Check every corner of i against every edge of k
    int inside = 0;
    bool overlap = false;
    unsigned int k_next;

    for(unsigned int k = 0; k<submap_i_corners.size(); k++){
        // Four corners
        k_next = k + 1;
        k_next = (k_next == submap_i_corners.size())? 0: k_next;
        // Check against four edges
        if(pointToLine(submap_i_corners.at(k), submap_i_corners.at(k_next), sift_i)){
            inside++;
        }
        else{
            break;
        }
    }
    overlap = (inside == 4)? true: false;
    return overlap;
}

class submapConstructor
{
    typedef std::tuple<sensor_msgs::PointCloud2Ptr, tf::Transform> ping_raw;

public:
    
    submapConstructor(std::string node_name, ros::NodeHandle &nh);

    void siftMapCB(const sensor_msgs::PointCloud2Ptr &sift_map);

    void enableCB(const std_msgs::BoolPtr &enable_msg);

    void pingCB(const sensor_msgs::PointCloud2Ptr &mbes_ping,
                                   const nav_msgs::OdometryPtr &odom_msg);

    void addSubmap(std::vector<ping_raw> submap_pings);

    PointCloudT::Ptr extractLandmarksUnknown(SubmapObj& submap_i);

    void extractLandmarksKnown(SubmapObj& submap_i, 
                               PointCloudT::Ptr& landmarks, std::vector<int>& lm_idx);

    void checkForLoopClosures(SubmapObj& submap_i);
        
    std::string node_name_;
    ros::NodeHandle *nh_;
    ros::Publisher submaps_pub_, lm_idx_pub_;
    ros::Subscriber enable_subs_;
    ros::Subscriber sift_map_subs_;
    std::string map_frame_, odom_frame_, base_frame_, mbes_frame_;
    ros::ServiceServer synch_service_;

    message_filters::Subscriber<sensor_msgs::PointCloud2> mbes_subs_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_subs_;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> *synch_;

    SubmapsVec submaps_vec_;
    std::vector<ping_raw> submap_raw_;
    int submaps_cnt_;

    PointCloudT::Ptr sift_map_;
    bool first_sift_map_;
    bool known_association_;

    // std::vector<tf::Transform> tf_submaps_vec_;
    tf::TransformBroadcaster submaps_bc_;
    tf::TransformListener tflistener_;
    tf::StampedTransform tf_base_mbes_, tf_map_odom_;
    tf2_ros::StaticTransformBroadcaster static_broadcaster_;

    boost::shared_ptr<SubmapRegistration> gicp_reg_;
};

#endif