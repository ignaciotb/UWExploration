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

#include <pcl_ros/transforms.h>

#include <actionlib/server/simple_action_server.h>
#include <slam_msgs/MinibatchTrainingAction.h>
#include <slam_msgs/MinibatchTrainingGoal.h>
#include <slam_msgs/MinibatchTrainingResult.h>

using namespace std;

class RbpfSlam
{

public:
    RbpfSlam(string node_name, ros::NodeHandle &nh)
    {
        ros::NodeHandle *nh_;

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

        // Transforms from auv_2_ros
        try
        {
            ROS_DEBUG("Waiting for transforms");

            tf::StampedTransform mbes_tf;
            tf::StampedTransform m2o_tf;
            tfBuffer.waitForTransform(base_frame, mbes_frame, ros::Time(0), ros::Duration(10.0));
            tfBuffer.lookupTransform(base_frame, mbes_frame, ros::Time(0), mbes_tf);
            tfBuffer.waitForTransform(map_frame, odom_frame, ros::Time(0), ros::Duration(10.0));
            tfBuffer.lookupTransform(map_frame, odom_frame, ros::Time(0), m2o_tf);

            Eigen::Matrix4f base2mbes_mat;
            Eigen::Matrix4f m2o_mat;
            pcl_ros::transformAsMatrix(mbes_tf, base2mbes_mat);
            pcl_ros::transformAsMatrix(m2o_tf, m2o_mat);

            ROS_DEBUG("Transforms locked - RBPF node");
        }

        catch(const std::exception& e)
        {
            ROS_DEBUG("ERROR: Could not lookup transform from base_link to mbes_link");
        }

        // Initialize list of particles
        // TODO, need to create the Particle class first
        /* 
        .
        .
        .
        */

       // Subscription to the end of mission topic
        string finished_top;
        nh_->param<string>(("~survey_finished_top"), finished_top, "/survey_finished");
        ros::Subscriber finished_sub = nh.subscribe(finished_top, 100, synch_cb);
        bool survey_finished = false;

        // Start timing now
        float time = ros::Time::now().toSec();
        float old_time = ros::Time::now().toSec();

        // For Loop Closure detection
        bool lc_detected = false;

        // Main timer for the RBPF
        float rbpf_period;
        nh_->param<float>(("~rbpf_period"), rbpf_period, 0.3);
        ros::Timer timer = nh.createTimer(ros::Duration(rbpf_period), rbpf_update, false);

        // Subscription to real mbes pings
        string lc_manual_topic;
        nh_->param<string>(("~lc_manual_topic"), lc_manual_topic, "manual_lc");
        ros::Subscriber lc_manual_sub = nh.subscribe(lc_manual_topic, 1, manual_lc);

        // Empty service to synch the applications waiting for this node to start
        ROS_DEBUG("RBPF successfully instantiated");    

        string synch_top;
        nh_->param<string>(("~synch_topic"), synch_top, "/pf_synch");   
        ros::ServiceServer srv_server = nh.advertiseService(synch_top, empty_srv);

        // Service for sending minibatches of beams to the SVGP particles
        string mb_gp_name;
        nh_->param<string>(("~minibatch_gp_server"), mb_gp_name);
    	actionlib::SimpleActionServer<slam_msgs::MinibatchTrainingAction> server(nh_, mb_gp_name, boost::bind(&mb_cb, _1, &server), false);
        server.start();

        // The mission waypoints as a path
        string path_topic;
        nh_->param<string>(("~path_topic"), path_topic);
        ros::Subscriber path_sub = nh.subscribe(path_topic, 1, path_cb);

        // Publisher for inducing points to SVGP maps
        string ip_top;
        nh_->param<string>(("~inducing_points_top"), ip_top);
        ros::Publisher ip_pub = nh.advertise<sensor_msgs::PointCloud2>(ip_top, 1);

        // Publisher for particles indexes to be resamples
        string p_resampling_top;
        nh_->param<string>(("~p_resampling_top"), p_resampling_top);
        ros::Publisher p_resampling_pub = nh.advertise<std_msgs::Float32>(p_resampling_top, 10);

        ros::spin();
    }

};
