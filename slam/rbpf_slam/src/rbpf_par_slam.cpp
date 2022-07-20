#include "rbpf_par_slam.h"

RbpfSlam::RbpfSlam(ros::NodeHandle &nh, ros::NodeHandle &nh_mb) : nh_(&nh), nh_mb_(&nh_mb), rng_((std::random_device())()), g_((std::random_device())())
{
    // Get parameters from launch file
    nh_->param<int>(("particle_count"), pc_, 10);
    nh_->param<int>(("n_beams_mbes"), beams_real_, 512);
    nh_->param<string>(("map_frame"), map_frame_, "map");
    nh_->param<string>(("mbes_link"), mbes_frame_, "mbes_link");
    nh_->param<string>(("base_link"), base_frame_, "base_link");
    nh_->param<string>(("odom_frame"), odom_frame_, "odom");

    // Read covariance values
    nh_->param<float>(("measurement_std"), meas_std_, 0.01);
    nh_->param("init_covariance", init_cov_, vector<float>());
    nh_->param("resampling_noise_covariance", res_noise_cov_, vector<float>());
    nh_->param("motion_covariance", motion_cov_, vector<float>());

    // fill_n(n_eff_mask_, 3, pc_);
    // float pw_[pc_];
    // fill_n(pw_, pc_, 1.e-50);
    n_eff_mask_.insert(n_eff_mask_.end(), 3, pc_);
    pw_.insert(pw_.end(), pc_, 1e-50);
    n_eff_filt_ = 0.;
    count_pings_ = 0;
    // count_training_ = 0;

    // TODO: these strings don't need to be global variables
    // Initialize particle poses publisher
    nh_->param<string>(("particle_poses_topic"), pose_array_top_, "/particle_poses");
    pf_pub_ = nh_->advertise<geometry_msgs::PoseArray>(pose_array_top_, 10);

    nh_->param<string>(("dr_pose_topic"), pose_dr_top_, "/dr_pose");
    dr_estimate_pub_ = nh_->advertise<geometry_msgs::PoseStamped>(pose_dr_top_, 10);

    // Initialize average of poses publisher
    nh_->param<string>(("average_pose_topic"), avg_pose_top_, "/average_pose");
    avg_pub_ = nh_->advertise<geometry_msgs::PoseWithCovarianceStamped>(avg_pose_top_, 10);

    // Expected meas of PF outcome at every time step
    nh_->param<string>(("average_mbes_topic"), pf_mbes_top_, "/avg_mbes");
    pf_mbes_pub_ = nh_->advertise<sensor_msgs::PointCloud2>(pf_mbes_top_, 1);

    nh_->param<string>(("pf_stats_top"), stats_top_, "stats");
    stats_ = nh_->advertise<std_msgs::Float32MultiArray>(stats_top_, 10);

    // Action server for interacting with the GP posterior
    nh_->param<string>(("manipulate_gp_server"), manipulate_gp_server_, "gp_sample_server");

    // Timer for end of mission: finish when no more odom is being received
    mission_finished_ = false;
    time_wo_motion_ = 5.;

    // Transforms from auv_2_ros
            
    tf2_ros::TransformListener tf_listener(tf_buffer_);
    try
    {
        ROS_DEBUG("Waiting for transforms");
        auto asynch_1 = std::async(std::launch::async, [this]
                                       { return tf_buffer_.lookupTransform(base_frame_, mbes_frame_,
                                                                           ros::Time(0), ros::Duration(60.)); });

        auto asynch_2 = std::async(std::launch::async, [this]
                                       { return tf_buffer_.lookupTransform(map_frame_, odom_frame_,
                                                                           ros::Time(0), ros::Duration(60.)); });

        tf::StampedTransform mbes_tf;
        geometry_msgs::TransformStamped tfmsg_mbes_base = asynch_1.get();
        tf::transformMsgToTF(tfmsg_mbes_base.transform, mbes_tf);
        pcl_ros::transformAsMatrix(mbes_tf, base2mbes_mat_);

        tf::StampedTransform m2o_tf;
        geometry_msgs::TransformStamped tfmsg_map_odom = asynch_2.get();
        tf::transformMsgToTF(tfmsg_map_odom.transform, m2o_tf);
        pcl_ros::transformAsMatrix(m2o_tf, m2o_mat_);

        ROS_INFO("Transforms locked - RBPF node");
    }
    catch (const std::exception &e)
    {
        ROS_ERROR("ERROR: Could not lookup transform from base_link to mbes_link");
    }

    // Subscription to the end of mission topic
    nh_->param<string>(("survey_finished_top"), finished_top_, "/survey_finished");
    finished_sub_ = nh_->subscribe(finished_top_, 100, &RbpfSlam::synch_cb, this);
    survey_finished_ = false;

    // Subscription to the save topic
    save_sub_ = nh_->subscribe("/save", 100, &RbpfSlam::save_cb, this);

    // Start timing now
    time_ = ros::Time::now().toSec();
    old_time_ = ros::Time::now().toSec();

    // Timer for the RBPF LC prompting
    nh_mb_->param<float>(("rbpf_period"), rbpf_period_, 0.3);
    timer_rbpf_ = nh_mb_->createTimer(ros::Duration(rbpf_period_), &RbpfSlam::rbpf_update, this, false);

    // Timer for updating RVIZ
    nh_->param<float>(("rviz_period"), rviz_period_, 0.3);
    if(rviz_period_ != 0.){
        timer_rviz_ = nh_->createTimer(ros::Duration(rviz_period_), &RbpfSlam::update_rviz, this, false);
    }

    // Subscription to manually triggering LC detection. Just for testing
    nh_->param<string>(("lc_manual_topic"), lc_manual_topic_, "/manual_lc");
    lc_manual_sub_ = nh_->subscribe(lc_manual_topic_, 1, &RbpfSlam::manual_lc, this);

    // The mission waypoints as a path
    nh_->param<string>(("path_topic"), path_topic_, "/waypoints");
    path_sub_ = nh_->subscribe(path_topic_, 1, &RbpfSlam::path_cb, this);

    // Publisher for inducing points to SVGP maps
    std::string ip_top;
    nh_->param<string>(("inducing_points_top"), ip_top, "/inducing_points");
    ip_pub_ = nh_->advertise<sensor_msgs::PointCloud2>(ip_top, 1);

    // Publisher for particles indexes to be resampled
    std::string p_resampling_top;
    nh_->param<string>(("gp_resampling_top"), p_resampling_top, "/resample_top");
    for (int i = 0; i < pc_; i++)
        p_resampling_srvs_.push_back(nh_->serviceClient<slam_msgs::Resample>(p_resampling_top + "/particle_" + std::to_string(i)));

    // Service for sending minibatches of beams to the SVGP particles
    std::string mb_gp_name;
    nh_mb_->param<string>(("minibatch_gp_server"), mb_gp_name, "minibatch_server");
    as_mb_ = new actionlib::SimpleActionServer<slam_msgs::MinibatchTrainingAction>(*nh_mb_, mb_gp_name, boost::bind(&RbpfSlam::mb_cb, this, _1), false);
    as_mb_->start();

    // Action clients for manipulating the GP posteriors
    for (int i = 0; i < pc_; i++)
    {
        actionlib::SimpleActionClient<slam_msgs::ManipulatePosteriorAction> *ac =
                    new actionlib::SimpleActionClient<slam_msgs::ManipulatePosteriorAction>("/particle_" + std::to_string(i)
                    + manipulate_gp_server_, true);
        while (!ac->waitForServer() && ros::ok())
        {
            std::cout << "Waiting for SVGP plot server "
                      << "/particle_" + std::to_string(i) + manipulate_gp_server_ << std::endl;
            ros::Duration(2).sleep();
        }
        p_manipulate_acs_.push_back(ac);
    }

    // Create vector of beams indexes per ping.
    // It can be done only once since we're making sure all pings have the
    // same num of beams
    for (int n = 0; n < beams_real_; n++)
    {
        beams_idx_.push_back(n);
    }

    avg_pose_.header.frame_id = odom_frame_;

    start_training_ = false;
    time_avg_ = 0;
    count_mb_cbs_ = 0;
    count_mbes_cbs_ = 0;
    lc_detected_ = false;
    ancestry_sizes_.push_back(0);

    // Subscription to real mbes pings
    nh_->param<string>(("mbes_pings_topic"), mbes_pings_top_, "mbes_pings");
    mbes_sub_ = nh_->subscribe(mbes_pings_top_, 10, &RbpfSlam::mbes_real_cb, this);

    // Establish subscription to odometry message (intentionally last)
    nh_->param<string>(("odometry_topic"), odom_top_, "odom");
    odom_sub_ = nh_->subscribe(odom_top_, 100, &RbpfSlam::odom_callback, this);

    // Initialize the particles on top of LoLo 
    tf::StampedTransform o2b_tf;
    tfListener_.waitForTransform(odom_frame_, base_frame_, ros::Time(0), ros::Duration(30.0));
    tfListener_.lookupTransform(odom_frame_, base_frame_, ros::Time(0), o2b_tf);
    double x, y, z, roll_o2b, pitch_o2b, yaw_o2b;
    x = o2b_tf.getOrigin().x();
    y = o2b_tf.getOrigin().y();
    z = o2b_tf.getOrigin().z();
    o2b_tf.getBasis().getRPY(roll_o2b, pitch_o2b, yaw_o2b);
    init_p_pose_(0)= x;
    init_p_pose_(1)= y;
    init_p_pose_(2)= z;
    init_p_pose_(3)= roll_o2b;
    init_p_pose_(4)= pitch_o2b;
    init_p_pose_(5)= yaw_o2b;

    // Create one particle on top of the GT vehicle pose. Only for testing
    // for (int i=0; i<1; i++){
    //     particles_.emplace_back(RbpfParticle(beams_real_, pc_, i, base2mbes_mat_, m2o_mat_, init_p_pose_,
    //                                        std::vector<float>(6, 0.), meas_std_, motion_cov_));
    // }
    
    // Create particles
    for (int i=0; i<pc_; i++){
        particles_.emplace_back(RbpfParticle(beams_real_, pc_, i, base2mbes_mat_, m2o_mat_, init_p_pose_,
                                            init_cov_, meas_std_, motion_cov_));
    }

    // Dead reckoning particle
    dr_particle_.emplace_back(RbpfParticle(beams_real_, pc_, pc_ + 1, base2mbes_mat_, m2o_mat_, init_p_pose_,
                                           std::vector<float>(6, 0.), meas_std_, std::vector<float>(6,0.)));

    // Initialize aux variables
    ping_mat_ = Eigen::MatrixXf(beams_real_, 3);
    int mb_size;
    nh_->param<int>(("svgp_minibatch_size"), mb_size, 100);
    mb_mat_ = Eigen::MatrixXf(mb_size, 3);

    std::string enable_lc_top;
    nh_->param<string>(("particle_enable_lc"), enable_lc_top, "/enable_lc");
    enable_lc_sub_ = nh_->subscribe(enable_lc_top, 100, &RbpfSlam::enable_lc, this);

    ROS_INFO("RBPF instantiated");
}

bool RbpfSlam::empty_srv(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
{
    ROS_DEBUG("RBPF ready");
    return true;
}

void RbpfSlam::manual_lc(const std_msgs::Bool::ConstPtr& lc_msg) { lc_detected_ = true; }


// Receive IDs of SVGPs when they are ready to be used for LC prompting 
void RbpfSlam::enable_lc(const std_msgs::Int32::ConstPtr& enable_lc)
{
    svgp_lc_ready_.push_back(enable_lc->data);
}

void RbpfSlam::path_cb(const nav_msgs::PathConstPtr& wp_path)
{

    if (wp_path->poses.size() > 0)
    {
        if (!start_training_){
            ROS_INFO("Sending inducing points");
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
            start_training_ = true; // We can start to check for loop closures
            ip_pub_.publish(ip_pcloud);
            // This service will start the auv simulation or auv_2_ros nodes to start the mission
            nh_->param<string>(("synch_topic"), synch_top_, "/pf_synch");
            srv_server_ = nh_->advertiseService(synch_top_, &RbpfSlam::empty_srv, this);
        }
    }
    else{
        ROS_WARN("Received empty mission");
    }
}


void RbpfSlam::synch_cb(const std_msgs::Bool::ConstPtr& finished_msg)
{
    ROS_INFO("RBPF node: Survey finished received");
    mission_finished_ = true;
    save_gps(false);
    ROS_INFO("We done bitches");
}


void RbpfSlam::save_cb(const std_msgs::Bool::ConstPtr& save_msg)
{
    ROS_INFO("PF node: saving without finishing the mission");
    save_gps(save_msg->data);
    ROS_INFO("Aaaand it's saved");
}

void RbpfSlam::mbes_real_cb(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    if (mission_finished_ != true && start_training_)
    {
        // Beams in vehicle mbes frame
        // Store in pings history
        auto t1 = high_resolution_clock::now();
        mbes_history_.emplace_back(Pointcloud2msgToEigen(*msg, beams_real_));

        // Store latest mbes msg for timing
        latest_mbes_ = *msg;

        pings_idx_.push_back(count_pings_);
        // std::cout << "Number of pings " << pings_idx_.size() << std::endl;
        count_pings_ += 1;

        // Store in history particles poses corresponding to the current ping
        this->update_particles_history();

        // auto t2 = high_resolution_clock::now();
        // duration<double, std::milli> ms_double = t2 - t1;
        // std::cout << ms_double.count() / 1000.0 << std::endl;
        // count_mbes_cbs_++;
        // time_avg_ += ms_double.count();
        // std::cout << (time_avg_ / double(count_mb_cbs_))/1000.0 << std::endl;
        // std::cout << count_mb_cbs_ << std::endl;
    }
}

void RbpfSlam::rbpf_update(const ros::TimerEvent&)
{
    if(!mission_finished_)
    {
        if(latest_mbes_.header.stamp > prev_mbes_.header.stamp)
        {
            prev_mbes_ = latest_mbes_;
            // Conditions to start LC prompting:
            // 1) Pings collected > 1000: prevents from sampling undertrained GPs
            // 2) Num of GPs whose ELBO has converged > Num particles/2
            if(count_pings_ > 1000 && svgp_lc_ready_.size() > std::round(pc_ * 9/10)){
                this->update_particles_weights(latest_mbes_, odom_latest_);
            }
        }
    }
}

void RbpfSlam::odom_callback(const nav_msgs::OdometryConstPtr& odom_msg)
{
    time_ = odom_msg->header.stamp.toSec();
    odom_latest_ = *odom_msg;

    // // Flag to finish the mission
    if(mission_finished_ != true && start_training_)
    {
        // Motion prediction
        if (time_ > old_time_)
        {
            // Added in the MBES CB to synch the DR steps with the pings log
            nav_msgs::Odometry odom_cp = odom_latest_; // local copy
            float dt = float(time_ - old_time_);
            this->predict(odom_cp, dt);
        }
    }
    old_time_ = time_;
    
    // // Update stats and visual
    // publish_stats(*odom_msg);
    // update_rviz();
}

void RbpfSlam::mb_cb(const slam_msgs::MinibatchTrainingGoalConstPtr& goal)
{
    int pc_id = goal->particle_id;

    // Randomly pick mb_size/beams_per_ping pings
    int mb_size = goal->mb_size;
    int beams_per_ping = 10;
    slam_msgs::MinibatchTrainingResult result;
    Eigen::Vector3f pos_i;
    Eigen::Matrix3f rot_i;
    int ping_i;
    // If enough beams collected to start minibatch training
    // and survey not finished
    if (mission_finished_ != true && count_pings_ > (mb_size / beams_per_ping))
    {
        auto t1 = high_resolution_clock::now();
        // Shuffle indexes of pings collected so far and take the first int(mb_size / beams_per_pings)
        std::shuffle(pings_idx_.begin(), pings_idx_.end(), g_);
        for (int i = 0; i < int(mb_size / beams_per_ping); i++)
        {
            // Take the first 10 pings from the latest collected
            if(i < 10){
                ping_i = pings_idx_.size() - (i*2+2);    
            }
            // And take the rest randomly from the history
            else{
                ping_i = pings_idx_.at(i);
            }
            // Avoid using the latest ping since some particle might not have updated their pose histories
            // std::cout << "Current ping " << ping_i << std::endl;
            if(ping_i == pings_idx_.size()-1){
                ping_i = pings_idx_.size() - 2;
            }
            auto ancestry_it = ancestry_sizes_.begin();
            ancestry_it = std::find_if(ancestry_it, ancestry_sizes_.end(), [&](const int &ancestry_size_i)
                                       { return ping_i < ancestry_size_i; });
            int index;
            if (ancestry_it == ancestry_sizes_.begin())
            {
                index = 0;
            }
            else if (ancestry_it == ancestry_sizes_.end())
            {
                if (ancestry_sizes_.size() == 1)
                {
                    index = 0;
                }
                else
                {
                    index = ancestry_sizes_.size() - 1;
                }
            }
            else
            {
                index = std::distance(ancestry_sizes_.begin(), ancestry_it) - 1;
            }

            // std::cout << "Number of pings " << pings_idx_.size() << std::endl;
            // std::cout << "Ancestry index " << index << std::endl;
            // std::cout << "Ancestry sizes ";
            // for(int i = 0; i< ancestry_sizes_.size(); i++){
            //     std::cout << ancestry_sizes_.at(i) << " ";
            // }
            // std::cout << std::endl;

            // This check will make sure the ping_i corresponds to a DR step that hasn't been
            // updated yet by the motion_prediction()
            {
                std::lock_guard<std::mutex> lock(*particles_.at(pc_id).pc_mutex_);
                int idx_ping = ping_i - ancestry_sizes_.at(index);
                if (idx_ping < particles_.at(pc_id).pos_history_.at(index)->size()){
                    // Transform x random beams to particle pose in map frame
                    pos_i = particles_.at(pc_id).pos_history_.at(index)->at(idx_ping);
                    rot_i = particles_.at(pc_id).rot_history_.at(index)->at(idx_ping);
                    // Sample beams_per_pings beams from ping ping_i
                    std::shuffle(beams_idx_.begin(), beams_idx_.end(), g_);
                    for (int b = 0; b < beams_per_ping; b++)
                    {
                        mb_mat_.row(i * beams_per_ping + b) = (rot_i * mbes_history_.at(ping_i).row(beams_idx_.at(b)).transpose()
                                                                + pos_i).transpose();
                    }
                }
                // If ping_i out of index, take another one
                else{
                    ROS_DEBUG("MBES and DR histories out of synch ");
                    i--;
                }
            }
        }

        // TODO: can this be faster?
        // Parse into pointcloud2
        sensor_msgs::PointCloud2 mbes_pcloud;
        eigenToPointcloud2msg(mbes_pcloud, mb_mat_);
        // For testing only
        // mbes_pcloud.header.frame_id = "map";
        // pf_mbes_pub_.publish(mbes_pcloud);

        // Set action as success
        result.success = true;
        result.minibatch = mbes_pcloud;
        as_mb_->setSucceeded(result);

        /* For timing */
        // auto t2 = high_resolution_clock::now();
        // duration<double, std::milli> ms_double = t2 - t1;
        // time_avg_ += ms_double.count();
        // count_mb_cbs_++;
        // std::cout << (time_avg_ / double(count_mb_cbs_))/1000.0 << std::endl;
        // std::cout << count_mb_cbs_ << std::endl;
    }

    // If not enough beams collected to start the minibatch training
    else
    {
        result.success = false;
        as_mb_->setSucceeded(result);
    }
}

void RbpfSlam::save_gps(const bool plot)
{
    updated_saved_ids_.clear();

    int pings_t = mbes_history_.size()-1;
    Eigen::Vector3f pos_i;
    Eigen::Matrix3f rot_i;
    slam_msgs::ManipulatePosteriorGoal goal;
    int ping_i;
    for(int p=0; p<pc_; p++)
    // for(int p=0; p<4; p++)
    {
        Eigen::MatrixXf mbes_mat(pings_t * beams_real_, 3);
        Eigen::MatrixXf track_position_mat(pings_t, 3);
        Eigen::MatrixXf track_orientation_mat(pings_t, 3);
        int rots = 0;
        for (int ping_i = 0; ping_i < pings_t; ping_i++)
        {
            auto ancestry_it = ancestry_sizes_.begin();
            ancestry_it = std::find_if(ancestry_it, ancestry_sizes_.end(), [&](const int &ancestry_size_i)
                                       { return ping_i < ancestry_size_i; });
            int index;
            if (ancestry_it == ancestry_sizes_.begin())
            {
                index = 0;
            }
            else if (ancestry_it == ancestry_sizes_.end())
            {
                if (ancestry_sizes_.size() == 1)
                {
                    index = 0;
                }
                else
                {
                    index = ancestry_sizes_.size() - 1;
                }
            }
            else
            {
                index = std::distance(ancestry_sizes_.begin(), ancestry_it) - 1;
            }
            // Transform x random beams to particle pose in map frame
            {
                std::lock_guard<std::mutex> lock(*particles_.at(p).pc_mutex_);
                int idx_ping = ping_i - ancestry_sizes_.at(index);
                if (idx_ping < particles_.at(p).pos_history_.at(index)->size())
                {
                    pos_i = particles_.at(p).pos_history_.at(index)->at(idx_ping);
                    rot_i = particles_.at(p).rot_history_.at(index)->at(idx_ping);
                    
                    // TODO: get rid of loop with colwise operations on mbes_history
                    for (int b = 0; b < mbes_history_.at(ping_i).rows(); b++)
                    {
                        mbes_mat.row(ping_i * beams_real_ + b) = (rot_i * mbes_history_.at(ping_i).row(b).transpose() + pos_i).transpose();
                    }
                    // Store particle position for that ping
                    track_position_mat.row(ping_i) = pos_i.transpose();
                    track_orientation_mat.row(ping_i) = rot_i.eulerAngles(0,1,2); 
                }
                else
                {
                    // TODO: how to handle this loss of data?
                    ROS_WARN("MBES and DR histories out of synch when saving GPs");
                }
            }
        }

        // Parse into pointcloud2
        sensor_msgs::PointCloud2 mbes_pcloud;
        eigenToPointcloud2msg(mbes_pcloud, mbes_mat);

        sensor_msgs::PointCloud2 track_position_pcloud;
        eigenToPointcloud2msg(track_position_pcloud, track_position_mat);

        sensor_msgs::PointCloud2 track_orientation_pcloud;
        eigenToPointcloud2msg(track_orientation_pcloud, track_orientation_mat);

        // Set action as success
        goal.pings = mbes_pcloud;
        goal.track_position = track_position_pcloud;
        goal.track_orientation = track_orientation_pcloud;
        goal.plot = plot;
        goal.sample = false;

        // We want these calls to be secuential since plotting is GPU-heavy
        // and not time critical, so we want the particles to do it one by one.
        if(plot){
            bool received = p_manipulate_acs_.at(p)->waitForResult();
        }
        else{ 
            p_manipulate_acs_.at(p)->sendGoal(goal, boost::bind(&RbpfSlam::saveCB, this, _1, _2));
        }
    }

    // Collect all updated weights and call resample()
    while (updated_saved_ids_.size() < pc_ && ros::ok())
    {
        // ROS_DEBUG("Waiting for weights");
        ros::Duration(1.).sleep();
        std::cout << "Number of SVGPs saved " << updated_saved_ids_.size() << std::endl;            
    }
    
}

void RbpfSlam::update_particles_weights(sensor_msgs::PointCloud2 &mbes_ping, nav_msgs::Odometry &odom)
{

    ROS_INFO("Updating weights");
    updated_w_ids_.clear();

    auto t1 = high_resolution_clock::now();
    // Latest ping depths in map frame
    Eigen::MatrixXf latest_mbes = Pointcloud2msgToEigen(mbes_ping, beams_real_);
    latest_mbes.rowwise() += Eigen::Vector3f(0, 0, m2o_mat_(2, 3) + odom.pose.pose.position.z).transpose();
    // Nacho: Fix for Lolo in the surface
    // latest_mbes_z_ = latest_mbes.col(2);
    latest_mbes_z_ = -latest_mbes.col(0);

    Eigen::Vector3f pos_i;
    Eigen::Matrix3f rot_i;
    slam_msgs::ManipulatePosteriorGoal goal;
    for(int p=0; p<pc_; p++){
        {
            // Latest particle pose
            std::lock_guard<std::mutex> lock(*particles_.at(p).pc_mutex_);
            pos_i = particles_.at(p).pos_history_.back()->back();
            rot_i = particles_.at(p).rot_history_.back()->back();
        }

        // TODO: get rid of loop with colwise operations on mbes_history
        for (int b = 0; b < mbes_history_.back().rows(); b++)
        {
            ping_mat_.row(b) = (rot_i * mbes_history_.back().row(b).transpose() + pos_i).transpose();
        }

        // Parse into pointcloud2
        sensor_msgs::PointCloud2 mbes_pcloud;
        eigenToPointcloud2msg(mbes_pcloud, ping_mat_);

        // Asynch goal
        goal.pings = mbes_pcloud;
        goal.sample = true;
        p_manipulate_acs_.at(p)->sendGoal(goal,
                                      boost::bind(&RbpfSlam::sampleCB, this, _1, _2));
    }

    // Collect all updated weights and call resample()
    int w_time_out = 0;
    while (updated_w_ids_.size() < pc_ && ros::ok() && w_time_out < rbpf_period_ * 100.)
    {
        // ROS_DEBUG("Waiting for weights");
        ros::Duration(0.01).sleep();
        w_time_out++;
    }

    // Safety timeout to handle lost goals on sampleCB()
    if (w_time_out < rbpf_period_ * 100.)
    {
        // Call resampling here and empty ids vector
        updated_w_ids_.clear();
        std::vector<double> weights;
        for(int i=0; i<pc_; i++){
            weights.push_back(particles_.at(i).w_);
        }
        this->resample(weights);
    }
    else{
        updated_w_ids_.clear();
        ROS_WARN("Lost weights on the way, skipping resampling");
    }

    // auto t2 = high_resolution_clock::now();
    // duration<double, std::milli> ms_double = t2 - t1;
    // std::cout << ms_double.count() / 1000.0 << std::endl;
}

void RbpfSlam::saveCB(const actionlib::SimpleClientGoalState &state,
                        const slam_msgs::ManipulatePosteriorResultConstPtr &result)
{
    updated_saved_ids_.push_back(result->p_id);
}

void RbpfSlam::sampleCB(const actionlib::SimpleClientGoalState &state,
                        const slam_msgs::ManipulatePosteriorResultConstPtr &result)
{
    std::vector<double> mu = result->mu;
    std::vector<double> sigma = result->sigma;

    Eigen::VectorXd mu_e = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(mu.data(), mu.size());
    Eigen::VectorXd sigma_e = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(sigma.data(), sigma.size());

    // TODO: pass sigma_e to particles here
    particles_.at(result->p_id).gp_covs_ = sigma_e;
    particles_.at(result->p_id).compute_weight(mu_e, latest_mbes_z_.cast<double>());
    // Particle p_id has computed its weight
    updated_w_ids_.push_back(result->p_id);
}

void RbpfSlam::predict(nav_msgs::Odometry odom_t, float dt)
{
    // Multithreading
    auto t1 = high_resolution_clock::now();
    Eigen::VectorXf noise_vec(6, 1);

    // Angular vel
    Eigen::Vector3f vel_rot = Eigen::Vector3f(odom_t.twist.twist.angular.x,
                                              odom_t.twist.twist.angular.y,
                                              odom_t.twist.twist.angular.z);

    // Linear vel
    Eigen::Vector3f vel_p = Eigen::Vector3f(odom_t.twist.twist.linear.x,
                                            odom_t.twist.twist.linear.y,
                                            odom_t.twist.twist.linear.z);

    // Depth (read directly)
    float depth = odom_t.pose.pose.position.z;

    for(int i = 0; i < pc_; i++)
    {
        pred_threads_vec_.emplace_back(std::thread(&RbpfParticle::motion_prediction, 
                                    &particles_.at(i), std::ref(vel_rot), std::ref(vel_p),
                                    depth, dt, std::ref(rng_)));
    }

    for (int i = 0; i < pc_; i++)
    {
        if (pred_threads_vec_[i].joinable())
        {
            pred_threads_vec_[i].join();
        }
    }
    pred_threads_vec_.clear();

    // Particle to compute DR without filtering
    dr_particle_.at(0).motion_prediction(vel_rot, vel_p, depth, dt, rng_);

    // auto t2 = high_resolution_clock::now();
    // duration<double, std::milli> ms_double = t2 - t1;
    // std::cout << ms_double.count() / 1000.0 << std::endl;
}

void RbpfSlam::update_particles_history()
{
    for(int i = 0; i < pc_; i++)
    {
        upd_threads_vec_.emplace_back(std::thread(&RbpfParticle::update_pose_history, 
                                    &particles_.at(i)));
    }

    for (int i = 0; i < pc_; i++)
    {
        if (upd_threads_vec_[i].joinable())
        {
            upd_threads_vec_[i].join();
        }
    }
    upd_threads_vec_.clear();
}

void RbpfSlam::update_rviz(const ros::TimerEvent &)
{
    if(start_training_)
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
            pose_i.orientation.x = q.x();
            pose_i.orientation.y = q.y();
            pose_i.orientation.z = q.z();
            pose_i.orientation.w = q.w();

            array_msg.poses.push_back(pose_i);
        }
        pf_pub_.publish(array_msg);
        average_pose(array_msg);

        // DR estimate
        geometry_msgs::PoseStamped pose_dr;
        pose_dr.header.frame_id = map_frame_;
        pose_dr.header.stamp = ros::Time::now();
        pose_dr.pose.position.x = dr_particle_.at(0).p_pose_(0);
        pose_dr.pose.position.y = dr_particle_.at(0).p_pose_(1);
        pose_dr.pose.position.z = dr_particle_.at(0).p_pose_(2);

        Eigen::AngleAxisf rollAngle(dr_particle_.at(0).p_pose_(3), Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf pitchAngle(dr_particle_.at(0).p_pose_(4), Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf yawAngle(dr_particle_.at(0).p_pose_(5), Eigen::Vector3f::UnitZ());
        Eigen::Quaternion<float> q = rollAngle * pitchAngle * yawAngle;
        pose_dr.pose.orientation.x = q.x();
        pose_dr.pose.orientation.y = q.y();
        pose_dr.pose.orientation.z = q.z();
        pose_dr.pose.orientation.w = q.w();
        dr_estimate_pub_.publish(pose_dr);

        // Update stats
        publish_stats(odom_latest_);
    }
}

void RbpfSlam::resample(vector<double> weights)
{
    int N_eff = pc_;
    vector<int> indices;
    vector<int> lost;
    vector<int> keep;
    vector<int> dupes;
    std::vector<int>::iterator idx;

    // std::cout << "Original weights :" << std::endl;
    // for(auto weight: weights){
    //     std::cout << weight << " ";
    // }
    // std::cout << std::endl;

    double w_sum = accumulate(weights.begin(), weights.end(), 0.0);
    if(w_sum == 0.)
        ROS_WARN("All weights are zero!");
    else
    {
        // Normalize weights
        transform(weights.begin(), weights.end(), weights.begin(), [&w_sum](auto& c){return c/w_sum;});
        std::cout << "Normalized weights :" << std::endl;
        for(auto weight: weights){
            std::cout << weight << " ";
        }
        std::cout << std::endl;

        // Compute effective number
        double w_sq_sum = inner_product(begin(weights), end(weights), begin(weights), 0.0); // Sum of squared elements of weigsth
        N_eff = 1 / w_sq_sum;
    }

    n_eff_mask_.erase(n_eff_mask_.begin());
    n_eff_mask_.push_back(N_eff);
    n_eff_filt_ = moving_average(n_eff_mask_, 3);

    std::cout << std::endl;
    std::cout << "Mask " << N_eff << " N_thres " << std::round(pc_ / 2) << std::endl;
    
    if (N_eff < std::round(pc_ / 2) || lc_detected_)
    // if (N_eff < 90 || lc_detected_)
    {
        // Resample particles
        ROS_INFO("Resampling");
        indices = systematic_resampling(weights);
        
        // For manual lc testing
        if(lc_detected_){
            indices = vector<int>(pc_, 0);
        }

        set<int> s(indices.begin(), indices.end());
        keep.assign(s.begin(), s.end());
        for(int i = 0; i < pc_; i++)
        {
            if (!count(keep.begin(), keep.end(), i))
                lost.push_back(i);
        }

        dupes = indices;
        for (int i : keep)
        {
            if (count(dupes.begin(), dupes.end(), i))
            {
                idx = find(dupes.begin(), dupes.end(), i);
                dupes.erase(idx);
            }
        }

        // Reasign and ddd noise to particles poses
        ROS_INFO("Reasigning poses");
        reassign_poses(lost, dupes);
        for(int i = 0; i < pc_; i++)
            particles_[i].add_noise(res_noise_cov_);

        // Reassign SVGP maps: send winning indexes to SVGP nodes
        slam_msgs::Resample k_ros;
        slam_msgs::Resample l_ros;
        std::cout << "Keep " << std::endl;
        auto t1 = high_resolution_clock::now();
        if(!dupes.empty())
        {
            for(int k : keep)
            {
                std::cout << k << " ";
                k_ros.request.p_id = k;
                p_resampling_srvs_[k].call(k_ros);
            }
            std::cout << std::endl;

            int j = 0;
            for (int l : lost)
            {
                // Send the ID of the particle to copy to the particle that has not been resampled
                l_ros.request.p_id = dupes[j];
                if(p_resampling_srvs_[l].call(l_ros)){
                    ROS_DEBUG("Dupe sent");
                }
                else{
                    ROS_WARN("Failed to call resample srv");
                }
                j++;
            }
        }
        lc_detected_ = false;

        // auto t2 = high_resolution_clock::now();
        // duration<double, std::milli> ms_double = t2 - t1;
        // std::cout << ms_double.count() / 1000.0 << std::endl;
    }
}

void RbpfSlam::reassign_poses(vector<int> lost, vector<int> dupes)
{
    // TODO: this can explode if not all the particles have the same size
    // Keep track of size of histories between resamples. Used for MB random sampling across whole history
    ancestry_sizes_.push_back(ancestry_sizes_.back() + particles_[0].pos_history_.back()->size());

    for(int i = 0; i < lost.size(); i++)
    {
        {
            // TODO: test this
            std::lock_guard<std::mutex> lock(*particles_.at(dupes[i]).pc_mutex_);
            // std::lock_guard<std::mutex> lock(particles_.at(lost[i]).pc_mutex_);
            // std::scoped_lock lock(particles_.at(dupes[i]).pc_mutex_, particles_.at(lost[i]).pc_mutex_);
            particles_[lost[i]].p_pose_ = particles_[dupes[i]].p_pose_;
            particles_[lost[i]].pos_history_ = particles_[dupes[i]].pos_history_;
            particles_[lost[i]].rot_history_ = particles_[dupes[i]].rot_history_;
        }
    }
    for(int i = 0; i < pc_; i++){
        // std::lock_guard<std::mutex> lock(particles_.at(i).pc_mutex_);
        particles_[i].pos_history_.emplace_back(std::shared_ptr<pos_track>(new pos_track()));
        particles_[i].rot_history_.emplace_back(std::shared_ptr<rot_track>(new rot_track()));
    }
}

vector<int> RbpfSlam::systematic_resampling(vector<double> weights)
{
    int N = weights.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0, 1);
    double rand_n = dis(gen);

    vector<int> range = arange(0, N, 1);
    vector<double> positions(N);
    vector<int> indexes(N, 0);
    vector<double> cum_sum(N);

    // make N subdivisions, and choose positions with a consistent random offset
    for(int i = 0; i < N; i++)
        positions[i] = (range[i] + rand_n) / double(N);

    partial_sum(weights.begin(), weights.end(), cum_sum.begin());

    int i = 0;
    int j = 0;

    while(i < N)
    {
        if(positions[i] < cum_sum[j])
        {
            indexes[i] = j;
            i++;
        }
        else
            j++;
    }
    return indexes;
}

void RbpfSlam::average_pose(geometry_msgs::PoseArray pose_list)
{
    vector<float> x;
    vector<float> y;
    vector<float> z;
    Eigen::Quaternionf q_avg = Eigen::Quaternionf(0., 0., 0., 0.);
    Eigen::MatrixXf Q_T(pose_list.poses.size(), 4);

    double roll_i, pitch_i, yaw_i;

    for (int i = 0; i < pose_list.poses.size(); i++)
    {
        x.push_back(pose_list.poses[i].position.x);
        y.push_back(pose_list.poses[i].position.y);
        z.push_back(pose_list.poses[i].position.z);
        
        Eigen::VectorXf v_e(4); 
        v_e << pose_list.poses[i].orientation.x,
                pose_list.poses[i].orientation.y,
                pose_list.poses[i].orientation.z,
                pose_list.poses[i].orientation.w;
        Q_T.row(i) = v_e.transpose();
    }

    // Compute averages
    float x_ave = accumulate(x.begin(), x.end(), 0.0) / pose_list.poses.size();
    float y_ave = accumulate(y.begin(), y.end(), 0.0) / pose_list.poses.size();
    float z_ave = accumulate(z.begin(), z.end(), 0.0) / pose_list.poses.size();

    // Avg of orientations as eigenvector of max eigenvalue of Q*Qt
    Eigen::MatrixXf Q_quad = Q_T.transpose() * Q_T;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXf> eig;
    eig.compute(Q_quad);
    // Eigen::Array<float, 1, Eigen::Dynamic> eigenvalues = eig.eigenvalues();
	Eigen::MatrixXf eigenvectors = eig.eigenvectors();

    avg_pose_.pose.pose.position.x = x_ave;
    avg_pose_.pose.pose.position.y = y_ave;
    avg_pose_.pose.pose.position.z = z_ave;
    avg_pose_.pose.pose.orientation.x = eigenvectors.col(3)[0];
    avg_pose_.pose.pose.orientation.y = eigenvectors.col(3)[1];
    avg_pose_.pose.pose.orientation.z = eigenvectors.col(3)[2];
    avg_pose_.pose.pose.orientation.w = eigenvectors.col(3)[3];

    avg_pose_.header.stamp = ros::Time::now();
    avg_pub_.publish(avg_pose_);

    // Compute covariance
    cov_ = Eigen::Matrix3f::Zero();
    vector<float> particle_pose;
    vector<float> average_pose;
    vector<float> dx;
    vector<float> dx_sq;
    Eigen::Matrix3f dx_sq_mat;

    for (int i = 0; i < pc_; i++)
    {
        particle_pose.push_back(x[i]);
        particle_pose.push_back(y[i]);
        particle_pose.push_back(z[i]);
        average_pose.push_back(avg_pose_.pose.pose.position.x);
        average_pose.push_back(avg_pose_.pose.pose.position.y);
        average_pose.push_back(avg_pose_.pose.pose.position.z);

        // dx = particle_pose - average_pose
        std::transform(particle_pose.begin(), particle_pose.end(), average_pose.begin(), std::back_inserter(dx),
                       [](float particle_pose, float average_pose)
                       { return particle_pose - average_pose; });

        // Create diagonal matrix from squared values of dx
        dx_sq = dx;
        transform(dx_sq.begin(), dx_sq.end(), dx_sq.begin(), [](float a){return a*a;});
        Eigen::VectorXf dx_sq_eigen = Eigen::Map<Eigen::VectorXf, Eigen::Unaligned>(dx_sq.data(), dx_sq.size());
        dx_sq_mat = Eigen::Matrix3f::Zero();
        dx_sq_mat.diagonal() = dx_sq_eigen.asDiagonal();

        cov_ += dx_sq_mat;
        cov_(0, 1) += dx[0]*dx[1];
        cov_(0, 2) += dx[0]*dx[2];
        cov_(1, 2) += dx[1]*dx[2];

        average_pose.clear();
        particle_pose.clear();
        dx.clear();
    }

    cov_ /= pc_;
}

void RbpfSlam::publish_stats(nav_msgs::Odometry gt_odom)
{
    // Send statistics for visualization
    std_msgs::Float32MultiArray stats;
    stats.data.push_back(n_eff_filt_);
    stats.data.push_back(pc_/2.f);
    stats.data.push_back(gt_odom.pose.pose.position.x);
    stats.data.push_back(gt_odom.pose.pose.position.y);
    stats.data.push_back(gt_odom.pose.pose.position.z);
    stats.data.push_back(avg_pose_.pose.pose.position.x);
    stats.data.push_back(avg_pose_.pose.pose.position.y);
    stats.data.push_back(avg_pose_.pose.pose.position.z);
    stats.data.push_back(dr_particle_.at(0).p_pose_(0));
    stats.data.push_back(dr_particle_.at(0).p_pose_(1));
    stats.data.push_back(dr_particle_.at(0).p_pose_(2));
    stats.data.push_back(cov_(0, 0));
    stats.data.push_back(cov_(0, 1));
    stats.data.push_back(cov_(0, 2));
    stats.data.push_back(cov_(1, 1));
    stats.data.push_back(cov_(1, 2));
    stats.data.push_back(cov_(2, 2));

    stats_.publish(stats);
    stats.data.clear();
}

float RbpfSlam::moving_average(vector<int> a, int n)
{
    vector<float> a_last(a.end() - n, a.end());
    float a_sum = accumulate(a_last.begin(), a_last.end(), 0);

    return (float)a_sum/n;
}

std::vector<int> RbpfSlam::arange(int start, int stop, int step)
{
    std::vector<int> values;
    for (int value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}

RbpfSlam::~RbpfSlam()
{
    delete (nh_);
    delete (nh_mb_);
}
