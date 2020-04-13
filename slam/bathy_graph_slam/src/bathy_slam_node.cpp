
#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <nav_msgs/Odometry.h>
#include <std_msgs/Bool.h>

#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <tf/transform_broadcaster.h>

#include "submaps_tools/submaps.hpp"

#include "data_tools/std_data.h"
#include "data_tools/benchmark.h"

#include <actionlib/client/simple_action_client.h>

#include <Eigen/Core>


class BathySlamNode{

    typedef std::tuple<sensor_msgs::PointCloud2Ptr, tf::Transform> ping_raw;

public:

    BathySlamNode(std::string node_name, ros::NodeHandle& nh):
    node_name_(node_name), nh_(&nh){

        std::string gt_pings_top, debug_pings_top, gt_odom_top, sim_pings_top, enable_top;
        nh_->param<std::string>("mbes_pings", gt_pings_top, "/gt/mbes_pings");
        nh_->param<std::string>("odom_gt", gt_odom_top, "/gt/odom");
        nh_->param<std::string>("map_frame", map_frame_, "map");
        nh_->param<std::string>("odom_frame", odom_frame_, "odom");
        nh_->param<std::string>("base_link", base_frame_, "base_frame");
        nh_->param<std::string>("mbes_link", mbes_frame_, "mbes_frame");
        nh_->param<std::string>("survey_finished_top", enable_top, "enable");

        mbes_subs_.subscribe(*nh_, gt_pings_top, 1);
        odom_subs_.subscribe(*nh_, gt_odom_top, 1);
        sync_ = new message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, nav_msgs::Odometry>
                (mbes_subs_, odom_subs_, 10);
        sync_->registerCallback(&BathySlamNode::pingCB, this);

        submaps_pub_ = nh_->advertise<sensor_msgs::PointCloud2>("/submaps", 10, false);
        enable_subs_ = nh_->subscribe(enable_top, 1, &BathySlamNode::enableCB, this);

        try {
            tflistener_.waitForTransform(mbes_frame_, base_frame_, ros::Time(0), ros::Duration(10.0) );
            tflistener_.lookupTransform(mbes_frame_, base_frame_, ros::Time(0), tf_mbes_base_);
            ROS_INFO("Locked transform base --> sensor");
        }
        catch(tf::TransformException &exception) {
            ROS_ERROR("%s", exception.what());
            ros::Duration(1.0).sleep();
        }

        submaps_cnt_ = 0;
    }

    void bcMapSubmapsTF(const ros::TimerEvent&){

        int cnt_i = 0;
        tf::StampedTransform tf_map_submap_stp;
        geometry_msgs::TransformStamped msg_map_submap;
        for(tf::Transform& tf_measi_map: tf_submaps_vec_){
             tf_map_submap_stp = tf::StampedTransform(tf_measi_map,
                                                      ros::Time::now(),
                                                      map_frame_,
                                                      "submap_" + std::to_string(cnt_i));

             cnt_i += 1;
             tf::transformStampedTFToMsg(tf_map_submap_stp, msg_map_submap);
             submaps_bc_.sendTransform(msg_map_submap);
        }
    }

private:

    std::string node_name_;
    ros::NodeHandle* nh_;
    ros::Publisher submaps_pub_;
    ros::Subscriber enable_subs_;
    std::string map_frame_, odom_frame_, base_frame_, mbes_frame_;

    message_filters::Subscriber<sensor_msgs::PointCloud2> mbes_subs_;
    message_filters::Subscriber<nav_msgs::Odometry> odom_subs_;
    message_filters::TimeSynchronizer<sensor_msgs::PointCloud2, nav_msgs::Odometry>* sync_;

    SubmapsVec submaps_vec_;
    std::vector<ping_raw> submap_raw_;
    unsigned int submaps_cnt_;

    std::vector<tf::Transform> tf_submaps_vec_;
    tf::TransformBroadcaster submaps_bc_;
    tf::TransformListener tflistener_;
    tf::StampedTransform tf_mbes_base_;

    void enableCB(const std_msgs::BoolPtr& enable_msg){
        if(enable_msg->data == true){
            ROS_INFO_STREAM("Creating benchmark");
            benchmark::track_error_benchmark benchmark("real_data");
            PointsT gt_map = pclToMatrixSubmap(submaps_vec_);
            PointsT gt_track = trackToMatrixSubmap(submaps_vec_);
            ROS_INFO_STREAM("About to... " << gt_track.size());
            benchmark.add_ground_truth(gt_map, gt_track);
        }
    }

    void pingCB(const sensor_msgs::PointCloud2Ptr& mbes_ping, const nav_msgs::OdometryPtr& odom_msg){

        tf::Transform ping_tf;
        tf::poseMsgToTF(odom_msg->pose.pose, ping_tf);
        submap_raw_.emplace_back(mbes_ping, ping_tf*tf_mbes_base_);

        if(submap_raw_.size()>100){
            this->submapConstructor();
            submap_raw_.clear();
        }
    }

    void submapConstructor(){
//        ROS_INFO_STREAM("Creating submap");

        // Store submap tf
        tf::Transform tf_submap_i = std::get<1>(submap_raw_.at(submap_raw_.size()/2));
        tf_submaps_vec_.push_back(tf_submap_i);

        // Create submap object
        SubmapObj submap_i;
        Eigen::Affine3d tf_affine;
        tf::transformTFToEigen(tf_submap_i, tf_affine);
        submap_i.submap_tf_ = tf_affine.matrix().cast<float>();

        for(ping_raw& ping_j: submap_raw_){
            PointCloudT pcl_ping;
            pcl::fromROSMsg(*std::get<0>(ping_j).get(), pcl_ping);
            pcl_ros::transformPointCloud(pcl_ping, pcl_ping, tf_submap_i.inverse() * std::get<1>(ping_j));
            submap_i.submap_pcl_ += pcl_ping;
        }
        submap_i.submap_pcl_.sensor_origin_ << submap_i.submap_tf_.translation();
        submap_i.submap_pcl_.sensor_orientation_ = submap_i.submap_tf_.linear();
        submap_i.submap_id_ = submaps_cnt_;
        submap_i.auv_tracks_.conservativeResize(submap_i.auv_tracks_.rows()+1, 3);
        submap_i.auv_tracks_.row(0) = submap_i.submap_tf_.translation().transpose().cast<double>();
        submaps_cnt_++;
        submaps_vec_.push_back(submap_i);

        // For RVIZ
        sensor_msgs::PointCloud2 submap_msg;
        pcl::toROSMsg(submap_i.submap_pcl_, submap_msg);
        submap_msg.header.frame_id = "submap_" + std::to_string(tf_submaps_vec_.size()-1);
        submap_msg.header.stamp = ros::Time::now();
        submaps_pub_.publish(submap_msg);
    }
};



int main(int argc, char** argv){

    ros::init(argc, argv, "bathy_slam_node");
    ros::NodeHandle nh("~");

    BathySlamNode* bathy_slam = new BathySlamNode(ros::this_node::getName(), nh);

    ros::Timer timer = nh.createTimer(ros::Duration(0.5), &BathySlamNode::bcMapSubmapsTF, bathy_slam);

    ros::spin();
    ros::waitForShutdown();

    if(!ros::ok()){
        delete bathy_slam;
    }
    ROS_INFO("Bathy SLAM node finished");

    return 0;
}
