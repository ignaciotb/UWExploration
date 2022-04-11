#include "rbpf_par_slam.h"

RbpfSlam::RbpfSlam(ros::NodeHandle &nh) : nh_(&nh)
{
    // Get parameters from launch file
    nh_->param<int>(("particle_count"), pc_, 10);
    nh_->param<int>(("num_beams_sim"), beams_num_, 20);
    nh_->param<int>(("n_beams_mbes"), beams_real_, 512);
    nh_->param<float>(("mbes_open_angle"), mbes_angle_, M_PI / 180. * 60.);
    nh_->param<string>(("map_frame"), map_frame_, "map");
    nh_->param<string>(("mbes_link"), mbes_frame_, "mbes_link");
    nh_->param<string>(("base_link"), base_frame_, "base_link");
    nh_->param<string>(("odom_frame"), odom_frame_, "odom");

    // Read covariance values
    nh_->param<float>(("measurement_std"), meas_std_, 0.01);
    nh_->param("init_covariance", init_cov_, vector<float>());
    nh_->param("resampling_noise_covariance", res_noise_cov_, vector<float>());
    nh_->param("motion_covariance", motion_cov_, vector<float>());

    fill_n(n_eff_mask_, 3, pc_);
    float pw_[pc_];
    fill_n(pw_, pc_, 1.e-50);
    n_eff_filt_ = 0.;
    count_pings_ = 0;
    count_training_ = 0;
    firstFit_ = true;
    one_time_ = true;
    time2resample_ = false;

    poses_.header.frame_id = odom_frame_;
    avg_pose_.header.frame_id = odom_frame_;

    targets_ = Eigen::ArrayXf::Zero(1);

    // For the ancestry tree
    observations_ = Eigen::ArrayXf::Zero(1, 3);
    mapping_ = Eigen::ArrayXf::Zero(1, 3);
    p_ID_ = 0;
    time4regression_ = false;
    n_from_ = 1;
    ctr_ = 0;

    // Nacho
    pings_since_training_ = 0;
    map_updates_ = 0;

    // Initialize particle poses publisher
    nh_->param<string>(("particle_poses_topic"), pose_array_top_, "/particle_poses");
    pf_pub_ = nh_->advertise<geometry_msgs::PoseArray>(pose_array_top_, 10);

    // Initialize average of poses publisher
    nh_->param<string>(("average_pose_topic"), avg_pose_top_, "/average_pose");
    avg_pub_ = nh_->advertise<geometry_msgs::PoseWithCovarianceStamped>(avg_pose_top_, 10);

    // Expected meas of PF outcome at every time step
    nh_->param<string>(("average_mbes_topic"), pf_mbes_top_, "/avg_mbes");
    pf_mbes_pub_ = nh_->advertise<sensor_msgs::PointCloud2>(pf_mbes_top_, 1);

    nh_->param<string>(("pf_stats_top"), stats_top_, "stats");
    stats_ = nh_->advertise<std_msgs::Float32>(stats_top_, 10);

    // nh_->param<string>(("particle_sim_mbes_topic"), mbes_pc_top, "/sim_mbes");

    // Action server for plotting the GP maps
    nh_->param<string>(("plot_gp_server"), plot_gp_server_, "gp_plot_server");
    nh_->param<string>(("sample_gp_server"), sample_gp_server_, "gp_sample_server");

    // Subscription to real mbes pings
    nh_->param<string>(("mbes_pings_topic"), mbes_pings_top_, "mbes_pings");
    mbes_sub_ = nh_->subscribe(mbes_pings_top_, 100, &RbpfSlam::mbes_real_cb, this);

    // Establish subscription to odometry message (intentionally last)
    nh_->param<string>(("odometry_topic"), odom_top_, "odom");
    odom_sub_ = nh_->subscribe(odom_top_, 100, &RbpfSlam::odom_callback, this);

    // Timer for end of mission: finish when no more odom is being received
    mission_finished_ = false;
    time_wo_motion_ = 5.;

    // Transforms from auv_2_ros
    try
    {
        ROS_DEBUG("Waiting for transforms");

        tfListener_.waitForTransform(base_frame_, mbes_frame_, ros::Time(0), ros::Duration(10.0));
        tfListener_.lookupTransform(base_frame_, mbes_frame_, ros::Time(0), mbes_tf_);
        tfListener_.waitForTransform(map_frame_, odom_frame_, ros::Time(0), ros::Duration(10.0));
        tfListener_.lookupTransform(map_frame_, odom_frame_, ros::Time(0), m2o_tf_);

        pcl_ros::transformAsMatrix(mbes_tf_, base2mbes_mat_);
        pcl_ros::transformAsMatrix(m2o_tf_, m2o_mat_);

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
    nh_->param<string>(("survey_finished_top"), finished_top_, "/survey_finished");
    finished_sub_ = nh_->subscribe(finished_top_, 100, &RbpfSlam::synch_cb, this);
    survey_finished_ = false;

    // Start timing now
    time_ = ros::Time::now().toSec();
    old_time_ = ros::Time::now().toSec();

    // Main timer for the RBPF
    nh_->param<float>(("rbpf_period"), rbpf_period_, 0.3);
    timer_ = nh_->createTimer(ros::Duration(rbpf_period_), &RbpfSlam::rbpf_update, this, false);

    // Subscription to manually triggering LC detection. Just for testing
    nh_->param<string>(("lc_manual_topic"), lc_manual_topic_, "/manual_lc");
    lc_manual_sub_ = nh_->subscribe(lc_manual_topic_, 1, &RbpfSlam::manual_lc, this);

    nh_->param<string>(("synch_topic"), synch_top_, "/pf_synch");
    srv_server_ = nh_->advertiseService(synch_top_, &RbpfSlam::empty_srv, this);


    // The mission waypoints as a path
    nh_->param<string>(("path_topic"), path_topic_, "/waypoints");
    path_sub_ = nh_->subscribe(path_topic_, 1, &RbpfSlam::path_cb, this);

    // // Publisher for inducing points to SVGP maps
    // nh_->param<string>(("inducing_points_top"), ip_top);
    // ip_pub = nh_->advertise<sensor_msgs::PointCloud2>(ip_top, 1);

    // // Publisher for particles indexes to be resamples
    // nh_->param<string>(("p_resampling_top"), p_resampling_top);
    // p_resampling_pub = nh_->advertise<std_msgs::Float32>(p_resampling_top, 10);
    
    // Service for sending minibatches of beams to the SVGP particles
    // TODO: MOVE THE DEFINITION OF THIS ACTION SERVER IN THE HEADER
    // nh_->param<string>(("minibatch_gp_server"), mb_gp_name);
    // as_mb_ = new actionlib::SimpleActionServer<slam_msgs::MinibatchTrainingAction>(*nh_, mb_gp_name, boost::bind(&RbpfSlam::mb_cb, this, _1), false);
    // as_mb_->start();

    // Action clients for plotting the GP posteriors
    // for (int i = 0; i < pc_; i++)
    // {
    //     actionlib::SimpleActionClient<slam_msgs::PlotPosteriorAction> ac("/particle_" + std::to_string(i) + plot_gp_server_, true);
    //     ac.waitForServer();
    //     p_plot_acs_.push_back(ac);
    // }

    // // Action clients for sampling the GP posteriors
    // for (int i = 0; i < pc_; i++)
    // {
    //     actionlib::SimpleActionClient<slam_msgs::SamplePosteriorAction> ac("/particle_" + std::to_string(i) + sample_gp_server_, true);
    //     ac.waitForServer();
    //     p_sample_acs_.push_back(ac);
    // }

    ROS_INFO("RBPF instantiated");
}

bool RbpfSlam::empty_srv(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
{
    ROS_DEBUG("RBPF ready");
    return true;
}

void RbpfSlam::manual_lc(const std_msgs::Bool::ConstPtr& lc_msg) { lc_detected_ = true; }


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

    ip_pcloud = pack_cloud(map_frame_, i_points);
    ROS_DEBUG("Sending inducing points");
    ip_pub_.publish(ip_pcloud);
}


void RbpfSlam::synch_cb(const std_msgs::Bool::ConstPtr& finished_msg)
{
    ROS_DEBUG("PF node: Survey finished received");
    mission_finished_ = true;
    plot_gp_maps();
    ROS_DEBUG("We done bitches, this time in c++");
}   


void RbpfSlam::mbes_real_cb(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    Eigen::ArrayXXf real_mbes_full(msg->row_step, 3);
    std::vector<int> idx;

    if (mission_finished_ != true)
    {
        // Beams in vehicle mbes frame
        real_mbes_full = pcloud2ranges_full(*msg);
        // Selecting only self.beams_num of beams in the ping
        idx = linspace(0, msg->row_step-1, beams_num_);
        // Store in pings history
        // TODO - DOUBLE FOR LOOP IS TEMPORARY, CAN'T FIND EIGEN'S VERSION OF :
        for(int i = 0; i < beams_num_; i++) 
        { 
            for(int j = 0; j < 3; j++) 
            {
                mbes_history_[count_pings_] = real_mbes_full(idx[i], j); 
            }
        }

        // Store latest mbes msg for timing
        latest_mbes_ = *msg;

        count_pings_++;

        // TODO - ONCE THE PARTICLE CLASS HAS BEEN CREATED
        // for (int i = 0; i < pc_; i++) { self.particles[i].ctr += 1 }

        pings_since_training_++;
    }
}

void RbpfSlam::rbpf_update(const ros::TimerEvent&)
{
    ROS_DEBUG("TODO");
}

void RbpfSlam::odom_callback(const nav_msgs::Odometry::ConstPtr& odom_msg)
{
    time_ = odom_msg->header.stamp.toSec();
    odom_latest_ = *odom_msg;

    // Flag to finish the mission
    if(mission_finished_ != true)
    {
        // Motion prediction
        if (time_ > old_time_) { predict(*odom_msg); }

        // Update stats and visual
        update_rviz();
        publish_stats(*odom_msg);
    }

    old_time_ = time_;

}

void RbpfSlam::mb_cb(const slam_msgs::MinibatchTrainingGoalConstPtr& goal)
{
    int pc_id = goal.particle_id;

    // Randomly pick mb_size/beams_per_ping pings 
    int mb_size = goal.mb_size;

    // If enough beams collected to start minibatch training
    if(mbes_history_.size() > mb_size/20)
    {
        int idx = 0;
    }
}

void RbpfSlam::plot_gp_maps()
{
    ROS_DEBUG("TODO");
}

void RbpfSlam::predict(nav_msgs::Odometry odom_t)
{
    float dt = time_ - old_time_;
    for(int i = 0; i < pc_; i++) 
    {
        ROS_DEBUG("TODO, AFTER THE PARTICLE CLASS");
    }
}

void RbpfSlam::update_rviz()
{
    ROS_DEBUG("TODO");
}

void RbpfSlam::publish_stats(nav_msgs::Odometry gt_odom)
{
    ROS_DEBUG("TODO");
}


RbpfSlam::~RbpfSlam(){
    delete(nh_);
}