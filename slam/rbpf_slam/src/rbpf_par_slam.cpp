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

    // TODO: these strings don't need to be global variables
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

        tfListener_.waitForTransform(base_frame_, mbes_frame_, ros::Time(0), ros::Duration(30.0));
        tfListener_.lookupTransform(base_frame_, mbes_frame_, ros::Time(0), mbes_tf_);
        tfListener_.waitForTransform(map_frame_, odom_frame_, ros::Time(0), ros::Duration(30.0));
        tfListener_.lookupTransform(map_frame_, odom_frame_, ros::Time(0), m2o_tf_);

        pcl_ros::transformAsMatrix(mbes_tf_, base2mbes_mat_);
        pcl_ros::transformAsMatrix(m2o_tf_, m2o_mat_);

        ROS_DEBUG("Transforms locked - RBPF node");
    }
    catch (const std::exception &e)
    {
        ROS_ERROR("ERROR: Could not lookup transform from base_link to mbes_link");
    }

    // Create particles
    // for (int i=0; i<pc_-1; i++){
    //     particles_.emplace_back(RbpfParticle(beams_num_, pc_, i, base2mbes_mat_, m2o_mat_,
    //                                         init_cov_, meas_std_, motion_cov_));
    // }
    // // Create one particle on top or the GT vehicle pose. Only for testing
    particles_.emplace_back(RbpfParticle(beams_num_, pc_, pc_, base2mbes_mat_, m2o_mat_,
                                        std::vector<float>(6, 0.), meas_std_, std::vector<float>(6, 0.)));

    // Subscription to the end of mission topic
    nh_->param<string>(("survey_finished_top"), finished_top_, "/survey_finished");
    finished_sub_ = nh_->subscribe(finished_top_, 100, &RbpfSlam::synch_cb, this);
    survey_finished_ = false;

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

    // Publisher for inducing points to SVGP maps
    std::string ip_top;
    nh_->param<string>(("inducing_points_top"), ip_top, "/inducing_points");
    ip_pub_ = nh_->advertise<sensor_msgs::PointCloud2>(ip_top, 1);

    // Publisher for particles indexes to be resamples
    std::string p_resampling_top;
    nh_->param<string>(("p_resampling_top"), p_resampling_top, "/resample_top");
    p_resampling_pub_ = nh_->advertise<std_msgs::Float32>(p_resampling_top, 10);
    
    // Service for sending minibatches of beams to the SVGP particles
    std::string mb_gp_name;
    nh_->param<string>(("minibatch_gp_server"), mb_gp_name, "minibatch_server");
    as_mb_ = new actionlib::SimpleActionServer<slam_msgs::MinibatchTrainingAction>(*nh_, mb_gp_name, boost::bind(&RbpfSlam::mb_cb, this, _1), false);
    as_mb_->start();

    // Action clients for plotting the GP posteriors
    for (int i = 0; i < pc_; i++)
    {
        actionlib::SimpleActionClient<slam_msgs::PlotPosteriorAction>* ac = 
                    new actionlib::SimpleActionClient<slam_msgs::PlotPosteriorAction>("/particle_" + std::to_string(i) + plot_gp_server_, true);
        while(!ac->waitForServer() && ros::ok())
        {
            std::cout << "Waiting for SVGP sample server "
                      << "/particle_" + std::to_string(i) + plot_gp_server_ << std::endl;
            ros::Duration(2).sleep();
        }
        p_plot_acs_.push_back(ac);
    }

    // Action clients for sampling the GP posteriors
    for (int i = 0; i < pc_; i++)
    {
        actionlib::SimpleActionClient<slam_msgs::SamplePosteriorAction> *ac = 
                    new actionlib::SimpleActionClient<slam_msgs::SamplePosteriorAction>("/particle_" + std::to_string(i) + sample_gp_server_, true);
        while (!ac->waitForServer() && ros::ok())
        {
            std::cout << "Waiting for SVGP plot server "
                      << "/particle_" + std::to_string(i) + sample_gp_server_ << std::endl;
            ros::Duration(2).sleep();
        }
        p_sample_acs_.push_back(ac);
    }

    // Start timing now
    time_ = ros::Time::now().toSec();
    old_time_ = ros::Time::now().toSec();

    ROS_INFO("RBPF instantiated");

}

bool RbpfSlam::empty_srv(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
{
    ROS_DEBUG("RBPF ready");
    return true;
}

void RbpfSlam::manual_lc(const std_msgs::Bool::ConstPtr& lc_msg) { lc_detected_ = true; }


void RbpfSlam::path_cb(const nav_msgs::PathConstPtr& wp_path)
{
    if (wp_path->poses.size() > 0){
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
        ROS_INFO("Sending inducing points");
        ip_pub_.publish(ip_pcloud);
    }
    else{
        ROS_WARN("Received empty mission");
    }
}


void RbpfSlam::synch_cb(const std_msgs::Bool::ConstPtr& finished_msg)
{
    ROS_DEBUG("PF node: Survey finished received");
    mission_finished_ = true;
    plot_gp_maps();
    ROS_DEBUG("We done bitches, this time in c++");
}   


void RbpfSlam::mbes_real_cb(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    if (mission_finished_ != true)
    {
        // Beams in vehicle mbes frame
        // Store in pings history
        mbes_history_.emplace_back(pcloud2ranges_full(*msg, beams_num_));

        // Store latest mbes msg for timing
        latest_mbes_ = *msg;

        pings_idx_.push_back(count_pings_);
        count_pings_ += 1;
    }
}

void RbpfSlam::rbpf_update(const ros::TimerEvent&)
{
    ROS_DEBUG("TODO");
}

void RbpfSlam::odom_callback(const nav_msgs::OdometryConstPtr& odom_msg)
{
    time_ = odom_msg->header.stamp.toSec();
    odom_latest_ = *odom_msg;

    // Flag to finish the mission
    if(mission_finished_ != true)
    {
        // Motion prediction
        if (time_ > old_time_) 
        { 
            this->predict(*odom_msg); 
        }
        // Update stats and visual
        update_rviz();
        // publish_stats(*odom_msg);
    }
    old_time_ = time_;

}

void RbpfSlam::mb_cb(const slam_msgs::MinibatchTrainingGoalConstPtr& goal)
{
    int pc_id = goal->particle_id;

    // Randomly pick mb_size/beams_per_ping pings 
    int mb_size = goal->mb_size;

    sensor_msgs::PointCloud2 mbes_pcloud;
    std::random_device rd;
    std::mt19937 g(rd());

    int beams_per_pings = 20;
    Eigen::MatrixXf mb_mat(mb_size, 3);

    // If enough beams collected to start minibatch training
    if (count_pings_ > (mb_size / beams_per_pings))
    {

        // Shuffle indexes of pings collected so far and take the first int(mb_size / beams_per_pings)
        std::shuffle(pings_idx_.begin(), pings_idx_.end()-1, g);
        for (int i = 0; i < int(mb_size / beams_per_pings); i++)
        // for (int i = count_pings_ - (mb_size / beams_per_pings); i < count_pings_-1; i++)
        {
            int ping_i = pings_idx_.at(i);
            // Transform x random beams to particle pose in map frame
            Eigen::Vector3f pos_i = particles_.at(pc_id).pos_history_.at(ping_i);
            Eigen::Matrix3f rot_i = particles_.at(pc_id).rot_history_.at(ping_i);

            // Sample beams_per_pings beams from ping ping_i
            std::vector<int> idx_beams;
            for (int n = 0; n < mbes_history_.at(ping_i).rows(); n++)
            {
                idx_beams.push_back(n);
            }
            std::shuffle(idx_beams.begin(), idx_beams.end(), g);

            for (int b = 0; b < beams_per_pings; b++)
            {
                mb_mat.row(i * beams_per_pings + b) = (rot_i * mbes_history_.at(ping_i).row(idx_beams.at(b)).transpose() + pos_i).transpose();
            }
        }

        // Parse into pointcloud2
        sensor_msgs::PointCloud2 mbes_pcloud;
        sensor_msgs::PointCloud2Modifier cloud_out_modifier(mbes_pcloud);
        cloud_out_modifier.setPointCloud2FieldsByString(1, "xyz");
        cloud_out_modifier.resize(mb_mat.rows());

        sensor_msgs::PointCloud2Iterator<float> iter_x(mbes_pcloud, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(mbes_pcloud, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(mbes_pcloud, "z");

        for (size_t i = 0; i < mb_mat.rows(); ++i, ++iter_x, ++iter_y, ++iter_z)
        {
            *iter_x = mb_mat.row(i)[0];
            *iter_y = mb_mat.row(i)[1];
            *iter_z = mb_mat.row(i)[2];
        }

        // Set action as success
        slam_msgs::MinibatchTrainingResult result;
        result.success = true;
        result.minibatch = mbes_pcloud;
        as_mb_->setSucceeded(result);
    }

    // If not enough beams collected to start the minibatch training
    else
    {
        slam_msgs::MinibatchTrainingResult result;
        result.success = false;
        as_mb_->setSucceeded(result);
    }
}


void RbpfSlam::plot_gp_maps()
{
    ROS_DEBUG("TODO");
}

void RbpfSlam::predict(nav_msgs::Odometry odom_t)
{
    float dt = float(time_ - old_time_);
    for(int i = 0; i < pc_; i++) 
    {
        particles_.at(i).motion_prediction(odom_t, dt);
        particles_.at(i).update_pose_history();
    }
}

void RbpfSlam::update_rviz()
{
    geometry_msgs::PoseArray array_msg;
    array_msg.header.frame_id = odom_frame_;
    array_msg.header.stamp = ros::Time::now();

    for (int i=0; i<pc_; i++){
        geometry_msgs::Pose pose_i;
        pose_i.position.x = particles_.at(i).p_pose_(0);
        pose_i.position.y = particles_.at(i).p_pose_(1);
        pose_i.position.z = particles_.at(i).p_pose_(2);

        Eigen::AngleAxisf rollAngle(particles_.at(i).p_pose_(3), Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf pitchAngle(particles_.at(i).p_pose_(4), Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf yawAngle(particles_.at(i).p_pose_(5), Eigen::Vector3f::UnitZ());
        Eigen::Quaternion<float> q = rollAngle * pitchAngle * yawAngle;
        // Nacho: check the order is correct
        pose_i.orientation.x = q.x();
        pose_i.orientation.y = q.y();
        pose_i.orientation.z = q.z();
        pose_i.orientation.w = q.w();

        array_msg.poses.push_back(pose_i);
    }
    pf_pub_.publish(array_msg);
    // TODO: add publisher avg pose from filter 
}

void RbpfSlam::publish_stats(nav_msgs::Odometry gt_odom)
{
    ROS_DEBUG("TODO");
}


RbpfSlam::~RbpfSlam(){
    delete(nh_);
}