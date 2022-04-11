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
#include <actionlib/client/simple_action_client.h>
#include <slam_msgs/MinibatchTrainingAction.h>
#include <slam_msgs/MinibatchTrainingGoal.h>
#include <slam_msgs/MinibatchTrainingResult.h>
#include <slam_msgs/SamplePosteriorAction.h>
#include <slam_msgs/PlotPosteriorAction.h>

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

    int pc_;
    int beams_num_;
    int beams_real_;
    float mbes_angle_;
    string map_frame_;
    string base_frame_;
    string mbes_frame_;
    string odom_frame_;

    tf::TransformListener tfListener_;

    // Loop closure
    bool lc_detected_;

    // Covariances
    float meas_std_;
    vector<float> init_cov_;
    vector<float> res_noise_cov_;
    vector<float> motion_cov_;

    // Global variables
    int n_eff_mask_[3];
    std::vector<Eigen::ArrayXXf, Eigen::aligned_allocator<Eigen::ArrayXXf>> mbes_history_;

    float n_eff_filt_;
    int count_pings_;
    int count_training_;
    bool firstFit_;
    bool one_time_;
    bool time2resample_;
    bool survey_finished_;
    float time_;
    float old_time_;
    float rbpf_period_;

    sensor_msgs::PointCloud2 prev_mbes_;
    sensor_msgs::PointCloud2 latest_mbes_;
    geometry_msgs::PoseArray poses_;
    geometry_msgs::PoseWithCovarianceStamped avg_pose_;

    Eigen::ArrayXf targets_;

    // Ancestry tree
    Eigen::ArrayXXf mapping_;
    Eigen::ArrayXXf observations_;
    std::vector<int> tree_list_;
    int p_ID_;
    bool time4regression_;
    int n_from_;
    int ctr_;

    // Nacho
    int pings_since_training_;
    int map_updates_;

    // Publishers
    ros::Publisher ip_pub_;
    ros::Publisher p_resampling_pub_;
    ros::Publisher pf_pub_;
    ros::Publisher avg_pub_;
    ros::Publisher pf_mbes_pub_;
    ros::Publisher stats_;
    string pose_array_top_;
    string avg_pose_top_;
    string pf_mbes_top_;
    string stats_top_;
    string mbes_pc_top_;
    string ip_top_;
    string p_resampling_top_;

    // Minibatch AS
    // actionlib::SimpleActionServer<slam_msgs::MinibatchTrainingAction>* as_mb_;

    // Action clients for sampling the GPs
    std::vector<actionlib::SimpleActionClient<slam_msgs::SamplePosteriorAction>> p_sample_acs_;

    // Action clients for plotting the GP posteriors
    std::vector<actionlib::SimpleActionClient<slam_msgs::PlotPosteriorAction>> p_plot_acs_;

    string plot_gp_server_;
    string sample_gp_server_;

    // Server
    ros::ServiceServer srv_server_;
    string synch_top_;
    string mb_gp_name_;

    // Subscribers
    ros::Subscriber mbes_sub_;
    ros::Subscriber odom_sub_;
    ros::Subscriber finished_sub_;
    ros::Subscriber lc_manual_sub_;
    ros::Subscriber path_sub_;

    string mbes_pings_top_;
    string odom_top_;
    string finished_top_;
    string lc_manual_topic_;
    string path_topic_;

    // End of mission timer
    bool mission_finished_;
    float time_wo_motion_;
    nav_msgs::Odometry odom_latest_;
    nav_msgs::Odometry odom_end_;

    // Transforms
    tf::StampedTransform mbes_tf_;
    tf::StampedTransform m2o_tf_;
    Eigen::Matrix4f base2mbes_mat_;
    Eigen::Matrix4f m2o_mat_;

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
    void predict(nav_msgs::Odometry odom_t);
    void update_rviz();
    void publish_stats(nav_msgs::Odometry gt_odom);
};
