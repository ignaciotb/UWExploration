// HEADER DEFINING THE RBPF_SLAM CLASS
#pragma once

#include "rbpf_particle.h"

// Standard dependencies
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>
#include <std_msgs/String.h>

#include <std_srvs/Empty.h>

#include <actionlib/server/simple_action_server.h>
#include <slam_msgs/MinibatchTrainingAction.h>
#include <slam_msgs/MinibatchTrainingGoal.h>
#include <slam_msgs/MinibatchTrainingResult.h>

using namespace std;

class RbpfSlam
{

public:
    RbpfSlam(ros::NodeHandle &nh);
    ~RbpfSlam();

private:    
    ros::NodeHandle *nh_;
    std::string node_name_;
    ros::Timer timer_;

    int pc;
    int beams_num;
    int beams_real;
    float mbes_angle;
    string map_frame;
    string base_frame;
    string mbes_frame;
    string odom_frame;

    tf::TransformListener tfListener;

    // Loop closure
    bool lc_detected;

    // Covariances
    float meas_std;
    vector<float> init_cov;
    vector<float> res_noise_cov;
    vector<float> motion_cov;

    // Global variables
    int n_eff_mask[3];
    std::vector<float> mbes_history;

    float n_eff_filt ;
    int count_pings;
    int count_training;
    bool firstFit;
    bool one_time;
    bool time2resample;
    bool survey_finished;
    float time;
    float old_time;
    float rbpf_period;

    sensor_msgs::PointCloud2 prev_mbes;
    sensor_msgs::PointCloud2 latest_mbes;
    geometry_msgs::PoseArray poses;
    geometry_msgs::PoseWithCovarianceStamped avg_pose;

    Eigen::ArrayXf targets;

    // Ancestry tree
    Eigen::ArrayXXf mapping;
    Eigen::ArrayXXf observations;
    std::vector<int> tree_list;
    int p_ID;
    bool time4regression;
    int n_from;
    int ctr;

    // Nacho
    int pings_since_training;
    int map_updates;

    // Publishers
    ros::Publisher ip_pub;
    ros::Publisher p_resampling_pub;
    ros::Publisher pf_pub;
    ros::Publisher avg_pub;
    ros::Publisher pf_mbes_pub;
    ros::Publisher stats;
    string pose_array_top;
    string avg_pose_top;
    string pf_mbes_top;
    string stats_top;
    string mbes_pc_top;
    string ip_top;
    string p_resampling_top;

    // Action servers
    // actionlib::SimpleActionServer<slam_msgs::MinibatchTrainingAction> server;

    string plot_gp_server;
    string sample_gp_server;

    // Server
    ros::ServiceServer srv_server;
    string synch_top;
    string mb_gp_name;

    // Subscribers
    ros::Subscriber mbes_sub;
    ros::Subscriber odom_sub;
    ros::Subscriber finished_sub;
    ros::Subscriber lc_manual_sub;
    ros::Subscriber path_sub;

    string mbes_pings_top;
    string odom_top;
    string finished_top;
    string lc_manual_topic;
    string path_topic;

    // Timer
    ros::Timer timer;
    // End of mission timer
    bool mission_finished;
    float time_wo_motion;
    nav_msgs::Odometry odom_latest;
    nav_msgs::Odometry odom_end;

    // Transforms
    tf::StampedTransform mbes_tf;
    tf::StampedTransform m2o_tf;
    Eigen::Matrix4f base2mbes_mat;
    Eigen::Matrix4f m2o_mat;

    // Callbacks
    bool empty_srv(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res);
    void mb_cb(const slam_msgs::MinibatchTrainingGoalConstPtr &goal);
    void manual_lc(const std_msgs::Bool::ConstPtr& lc_msg);
    void path_cb(const nav_msgs::Path::ConstPtr& wp_path);
    void synch_cb(const std_msgs::Bool::ConstPtr& finished_msg);
    void mbes_real_cb(const sensor_msgs::PointCloud2::ConstPtr& msg);
    void rbpf_update(const ros::TimerEvent&);
    void odom_callback(const nav_msgs::Odometry::ConstPtr& odom_msg);

    // Other functions
    void plot_gp_maps();
};
