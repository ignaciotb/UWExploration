#include "rbpf_par_slam.h"

RbpfSlam::RbpfSlam(ros::NodeHandle &nh) : nh_(&nh)
{
    // Get parameters from launch file
    nh_->param<int>(("particle_count"), pc, 10);
    nh_->param<int>(("num_beams_sim"), beams_num, 20);
    nh_->param<int>(("n_beams_mbes"), beams_real, 512);
    nh_->param<float>(("mbes_open_angle"), mbes_angle, M_PI / 180. * 60.);
    nh_->param<string>(("map_frame"), map_frame, "map");
    nh_->param<string>(("mbes_link"), mbes_frame, "mbes_link");
    nh_->param<string>(("base_link"), base_frame, "base_link");
    nh_->param<string>(("odom_frame"), odom_frame, "odom");

    // Read covariance values
    nh_->param<float>(("measurement_std"), meas_std, 0.01);
    nh_->param("init_covariance", init_cov, vector<float>());
    nh_->param("resampling_noise_covariance", res_noise_cov, vector<float>());
    nh_->param("motion_covariance", motion_cov, vector<float>());

    fill_n(n_eff_mask, 3, pc);
    float pw[pc];
    fill_n(pw, pc, 1.e-50);
    n_eff_filt = 0.;
    count_pings = 0;
    count_training = 0;
    firstFit = true;
    one_time = true;
    time2resample = false;

    poses.header.frame_id = odom_frame;
    avg_pose.header.frame_id = odom_frame;

    targets = Eigen::ArrayXf::Zero(1);

    // For the ancestry tree
    observations = Eigen::ArrayXf::Zero(1, 3);
    mapping = Eigen::ArrayXf::Zero(1, 3);
    p_ID = 0;
    time4regression = false;
    n_from = 1;
    ctr = 0;

    // Nacho
    pings_since_training = 0;
    map_updates = 0;

    // Initialize particle poses publisher
    nh_->param<string>(("particle_poses_topic"), pose_array_top, "/particle_poses");
    pf_pub = nh_->advertise<geometry_msgs::PoseArray>(pose_array_top, 10);

    // Initialize average of poses publisher
    nh_->param<string>(("average_pose_topic"), avg_pose_top, "/average_pose");
    avg_pub = nh_->advertise<geometry_msgs::PoseWithCovarianceStamped>(avg_pose_top, 10);

    // Expected meas of PF outcome at every time step
    nh_->param<string>(("average_mbes_topic"), pf_mbes_top, "/avg_mbes");
    pf_mbes_pub = nh_->advertise<sensor_msgs::PointCloud2>(pf_mbes_top, 1);

    nh_->param<string>(("pf_stats_top"), stats_top, "stats");
    stats = nh_->advertise<std_msgs::Float32>(stats_top, 10);

    // nh_->param<string>(("particle_sim_mbes_topic"), mbes_pc_top, "/sim_mbes");

    // Action server for plotting the GP maps
    nh_->param<string>(("plot_gp_server"), plot_gp_server, "gp_plot_server");
    nh_->param<string>(("sample_gp_server"), sample_gp_server, "gp_sample_server");

    // Subscription to real mbes pings
    nh_->param<string>(("mbes_pings_topic"), mbes_pings_top, "mbes_pings");
    mbes_sub = nh_->subscribe(mbes_pings_top, 100, &RbpfSlam::mbes_real_cb, this);

    // Establish subscription to odometry message (intentionally last)
    nh_->param<string>(("odometry_topic"), odom_top, "odom");
    odom_sub = nh_->subscribe(odom_top, 100, &RbpfSlam::odom_callback, this);

    // Timer for end of mission: finish when no more odom is being received
    mission_finished = false;
    time_wo_motion = 5.;

    // Transforms from auv_2_ros
    try
    {
        ROS_DEBUG("Waiting for transforms");

        tfListener.waitForTransform(base_frame, mbes_frame, ros::Time(0), ros::Duration(10.0));
        tfListener.lookupTransform(base_frame, mbes_frame, ros::Time(0), mbes_tf);
        tfListener.waitForTransform(map_frame, odom_frame, ros::Time(0), ros::Duration(10.0));
        tfListener.lookupTransform(map_frame, odom_frame, ros::Time(0), m2o_tf);

        pcl_ros::transformAsMatrix(mbes_tf, base2mbes_mat);
        pcl_ros::transformAsMatrix(m2o_tf, m2o_mat);

        ROS_DEBUG("Transforms locked - RBPF node");
    }

    catch (const std::exception &e)
    {
        ROS_ERROR("ERROR: Could not lookup transform from base_link to mbes_link");
    }

    // Initialize list of particles
    // TODO, need to create the Particle class first
    /* 
    .
    .
    .
    */

    // Subscription to the end of mission topic
    nh_->param<string>(("survey_finished_top"), finished_top, "/survey_finished");
    finished_sub = nh_->subscribe(finished_top, 100, &RbpfSlam::synch_cb, this);
    survey_finished = false;

    // Start timing now
    time = ros::Time::now().toSec();
    old_time = ros::Time::now().toSec();

    // Main timer for the RBPF
    nh_->param<float>(("rbpf_period"), rbpf_period, 0.3);
    timer = nh_->createTimer(ros::Duration(rbpf_period), &RbpfSlam::rbpf_update, this, false);

    // Subscription to manually triggering LC detection. Just for testing
    nh_->param<string>(("lc_manual_topic"), lc_manual_topic, "/manual_lc");
    lc_manual_sub = nh_->subscribe(lc_manual_topic, 1, &RbpfSlam::manual_lc, this);

    nh_->param<string>(("synch_topic"), synch_top, "/pf_synch");
    srv_server = nh_->advertiseService(synch_top, &RbpfSlam::empty_srv, this);

    // Service for sending minibatches of beams to the SVGP particles
    // TODO: MOVE THE DEFINITION OF THIS ACTION SERVER IN THE HEADER
    // nh_->param<string>(("minibatch_gp_server"), mb_gp_name);
    // actionlib::SimpleActionServer<slam_msgs::MinibatchTrainingAction> server(*nh_, mb_gp_name, boost::bind(&RbpfSlam::mb_cb, this, _1), false);
    // server.start();

    // The mission waypoints as a path
    nh_->param<string>(("path_topic"), path_topic, "/waypoints");
    path_sub = nh_->subscribe(path_topic, 1, &RbpfSlam::path_cb, this);

    // // Publisher for inducing points to SVGP maps
    // nh_->param<string>(("inducing_points_top"), ip_top);
    // ip_pub = nh_->advertise<sensor_msgs::PointCloud2>(ip_top, 1);

    // // Publisher for particles indexes to be resamples
    // nh_->param<string>(("p_resampling_top"), p_resampling_top);
    // p_resampling_pub = nh_->advertise<std_msgs::Float32>(p_resampling_top, 10);
    ROS_INFO("RBPF instantiated");
}

bool RbpfSlam::empty_srv(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
{
    ROS_DEBUG("RBPF ready");
    return true;
}

void RbpfSlam::manual_lc(const std_msgs::Bool::ConstPtr& lc_msg) { lc_detected = true; }

void RbpfSlam::path_cb(const nav_msgs::Path::ConstPtr& wp_path)
{
    ROS_INFO("Received path");
    std::vector<Eigen::RowVector3f> i_points;
    int wp_size = wp_path->poses.size();
    sensor_msgs::PointCloud2 ip_pcloud;
    Eigen::RowVector3f ip;
    
    auto poses = wp_path->poses;

    for (int i = 0; i < wp_size; i++)
    {
        ip << poses[i].pose.position.x, poses[i].pose.position.y, 0;
        i_points.push_back(ip);
    }

    ip_pcloud = pack_cloud(map_frame, i_points);
    ROS_DEBUG("Sending inducing points");
    ip_pub.publish(ip_pcloud);
}

void RbpfSlam::synch_cb(const std_msgs::Bool::ConstPtr& finished_msg)
{
    ROS_DEBUG("PF node: Survey finished received");
    mission_finished = true;
    plot_gp_maps();
    ROS_DEBUG("We done bitches, this time in c++");
}   

void RbpfSlam::mbes_real_cb(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    std::vector<geometry_msgs::Point32> real_mbes_full;
    if (mission_finished != true)
    {
        // Beams in vehicle mbes frame
        real_mbes_full = pcloud2ranges_full(*msg);
        // Selecting only self.beams_num of beams in the ping

        // Store in pings history
        // mbes_history.push_back(real_mbes_full[idx]);
    }
}

void RbpfSlam::rbpf_update(const ros::TimerEvent&)
{
    ROS_DEBUG("TODO");
}

void RbpfSlam::odom_callback(const nav_msgs::Odometry::ConstPtr& odom_msg)
{
    ROS_DEBUG("TODO");
}

void RbpfSlam::mb_cb(const slam_msgs::MinibatchTrainingGoalConstPtr& goal)
{
    ROS_DEBUG("TODO");
}

void RbpfSlam::plot_gp_maps()
{
    ROS_DEBUG("TODO");
}




RbpfSlam::~RbpfSlam(){
    delete(nh_);
}