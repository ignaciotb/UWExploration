// HEADER DEFINING THE RBPF_SLAM CLASS
#pragma once

// Standard dependencies
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <math.h>
#include <ros/ros.h>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
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

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>

#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>

using namespace std;

class RbpfSlam
{

public:
    RbpfSlam(string node_name, ros::NodeHandle &nh)
    {
        ros::NodeHandle *nh_;

        // nh_->param<double>((node_name_ + "/mhl_dist_mbes"), mh_dist_mbes, 0.2);
        // nh_->param<std::string>((node_name_ + "/pose_estimate_topic"), pose_topic, "/map_ekf");

        // Get parameters from rbpf_slam.launch
        int pc;
        int beams_num;
        int beams_real;
        float mbes_angle;
        string map_frame;
        string base_frame;
        string mbes_frame;
        string odom_frame;

        nh_->param<int>(("~particle_count"), pc, 10);
        nh_->param<int>(("~num_beams_sim"), beams_num, 20);
        nh_->param<int>(("~n_beams_mbes"), beams_real, 512);
        nh_->param<float>(("~mbes_open_angle"), mbes_angle, M_PI / 180. * 60.);
        nh_->param<string>(("~map_frame"), map_frame, "map");
        nh_->param<string>(("~mbes_link"), mbes_frame, "mbes_link");
        nh_->param<string>(("~base_link"), base_frame, "base_link");
        nh_->param<string>(("~odom_frame"), odom_frame, "odom");

        // Initialize tf listener
        tf::TransformListener tfBuffer;

        // Read covariance values
        float meas_std = 0.01;
        ros::param::get("~measurement_std", meas_std);

        vector<float> init_cov;
        nh_->param("~init_covariance", init_cov, vector<float>());

        vector<float> res_noise_cov;
        nh_->param("~resampling_noise_covariance", res_noise_cov, vector<float>());

        vector<float> motion_cov;
        nh_->param("~motion_covariance", motion_cov, vector<float>());

        // Global variables
        int n_eff_mask[3];
        fill_n(n_eff_mask, 3, pc);
        float pw[pc];
        fill_n(pw, pc, 1.e-50);
        list<float> mbes_history;

        float n_eff_filt = 0.;
        int count_pings = 0;
        int count_training = 0;
        bool firstFit = true;
        bool one_time = true;
        bool time2resample = false;

        sensor_msgs::PointCloud2 prev_mbes;
        sensor_msgs::PointCloud2 latest_mbes;
        geometry_msgs::PoseArray poses;
        geometry_msgs::PoseWithCovarianceStamped avg_pose;
        poses.header.frame_id = odom_frame;
        avg_pose.header.frame_id = odom_frame;

        Eigen::ArrayXf targets = Eigen::ArrayXf::Zero(1);


        // For the ancestry tree
        Eigen::ArrayXXf observations = Eigen::ArrayXf::Zero(1,3);
        Eigen::ArrayXXf mapping = Eigen::ArrayXf::Zero(1,3);
        list<int> tree_list;
        int p_ID = 0;
        bool time4regression = false;
        int n_from = 1;
        int ctr = 0;

        // Nacho
        int pings_since_training = 0;
        int map_updates = 0;

        // Initialize particle poses publisher
        string pose_array_top;
        nh_->param<string>(("~particle_poses_topic"), pose_array_top, "/particle_poses");
        ros::Publisher pf_pub = nh.advertise<geometry_msgs::PoseArray>(pose_array_top, 10);

        // Initialize average of poses publisher
        string avg_pose_top;
        nh_->param<string>(("~average_pose_topic"), avg_pose_top, "/average_pose");
        ros::Publisher avg_pub = nh.advertise<geometry_msgs::PoseWithCovarianceStamped>(avg_pose_top, 10);

        // Expected meas of PF outcome at every time step
        string pf_mbes_top;
        nh_->param<string>(("~average_mbes_topic"), pf_mbes_top, "/avg_mbes");
        ros::Publisher pf_mbes_pub = nh.advertise<sensor_msgs::PointCloud2>(pf_mbes_top, 1);

        string stats_top;
        nh_->param<string>(("~pf_stats_top"), stats_top, "stats");
        ros::Publisher stats = nh.advertise<std_msgs::Float32>(stats_top, 10);

        string mbes_pc_top;
        nh_->param<string>(("~particle_sim_mbes_topic"), mbes_pc_top, "/sim_mbes");

        // Action server for plotting the GP maps
        string plot_gp_server;
        string sample_gp_server;
        nh_->param<string>(("~plot_gp_server"), plot_gp_server, "gp_plot_server");
        nh_->param<string>(("~sample_gp_server"), sample_gp_server, "gp_sample_server");

        // Subscription to real mbes pings
        string mbes_pings_top;
        nh_->param<string>(("~mbes_pings_topic"), mbes_pings_top, "mbes_pings");
        ros::Subscriber mbes_sub = nh.subscribe(mbes_pings_top, 100, mbes_real_cb);

        // Establish subscription to odometry message (intentionally last)
        string odom_top;
        nh_->param<string>(("~odometry_topic"), odom_top, "odom");
        ros::Subscriber odom_sub = nh.subscribe(odom_top, 100, odom_callback);

        // Timer for end of mission: finish when no more odom is being received
        bool mission_finished = false;
        float time_wo_motion = 5.;
        nav_msgs::Odometry odom_latest;
        nav_msgs::Odometry odom_end;

    }

};
