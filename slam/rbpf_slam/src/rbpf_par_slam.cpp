#include "rbpf_par_slam.h"

RbpfSlam::RbpfSlam(ros::NodeHandle &nh, ros::NodeHandle &nh_mb) : nh_(&nh), nh_mb_(&nh_mb)
{
    // Get parameters from launch file
    nh_->param<int>(("particle_count"), pc_, 10);
    // nh_->param<int>(("num_beams_sim"), beams_num_, 20);
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

    // Initialize average of poses publisher
    nh_->param<string>(("average_pose_topic"), avg_pose_top_, "/average_pose");
    avg_pub_ = nh_->advertise<geometry_msgs::PoseWithCovarianceStamped>(avg_pose_top_, 10);

    // Expected meas of PF outcome at every time step
    nh_->param<string>(("average_mbes_topic"), pf_mbes_top_, "/avg_mbes");
    pf_mbes_pub_ = nh_->advertise<sensor_msgs::PointCloud2>(pf_mbes_top_, 1);

    nh_->param<string>(("pf_stats_top"), stats_top_, "stats");
    stats_ = nh_->advertise<std_msgs::Float32MultiArray>(stats_top_, 10);

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

        tf::StampedTransform mbes_tf;
        tf::StampedTransform m2o_tf;
        tfListener_.waitForTransform(base_frame_, mbes_frame_, ros::Time(0), ros::Duration(30.0));
        tfListener_.lookupTransform(base_frame_, mbes_frame_, ros::Time(0), mbes_tf);
        tfListener_.waitForTransform(map_frame_, odom_frame_, ros::Time(0), ros::Duration(30.0));
        tfListener_.lookupTransform(map_frame_, odom_frame_, ros::Time(0), m2o_tf);

        pcl_ros::transformAsMatrix(mbes_tf, base2mbes_mat_);
        pcl_ros::transformAsMatrix(m2o_tf, m2o_mat_);

        ROS_DEBUG("Transforms locked - RBPF node");
    }
    catch (const std::exception &e)
    {
        ROS_ERROR("ERROR: Could not lookup transform from base_link to mbes_link");
    }

    // // Create one particle on top or the GT vehicle pose. Only for testing
    particles_.emplace_back(RbpfParticle(beams_real_, pc_, 0, base2mbes_mat_, m2o_mat_,
                                        std::vector<float>(6, 0.), meas_std_, std::vector<float>(6, 0.)));
    // Create particles
    for (int i=1; i<pc_; i++){
        particles_.emplace_back(RbpfParticle(beams_real_, pc_, i, base2mbes_mat_, m2o_mat_,
                                            init_cov_, meas_std_, motion_cov_));
    }

    // Subscription to the end of mission topic
    nh_->param<string>(("survey_finished_top"), finished_top_, "/survey_finished");
    finished_sub_ = nh_->subscribe(finished_top_, 100, &RbpfSlam::synch_cb, this);
    survey_finished_ = false;

    // Main timer for the RBPF
    nh_->param<float>(("rbpf_period"), rbpf_period_, 0.3);
    timer_rbpf_ = nh_->createTimer(ros::Duration(rbpf_period_), &RbpfSlam::rbpf_update, this, false);

    nh_->param<float>(("rviz_period"), rviz_period_, 0.3);
    timer_rviz_ = nh_->createTimer(ros::Duration(rviz_period_), &RbpfSlam::update_rviz, this, false);

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

    // Create vector of beams indexes per ping.
    // It can be done only once since we're making sure all pings have the 
    // same num of beams
    for (int n = 0; n < beams_real_; n++)
    {
        beams_idx_.push_back(n);
    }
    
    avg_pose_.header.frame_id = odom_frame_;
    // Start timing now
    time_ = ros::Time::now().toSec();
    old_time_ = ros::Time::now().toSec();
    start_training_ = false;
    time_avg_ = 0;
    count_mb_cbs_ = 0;
    lc_detected_ = false;
    ancestry_sizes_.push_back(0);

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
        }
        // This service will start the auv simulation or auv_2_ros nodes to start the mission
        nh_->param<string>(("synch_topic"), synch_top_, "/pf_synch");
        srv_server_ = nh_->advertiseService(synch_top_, &RbpfSlam::empty_srv, this);
    }
    else{
        ROS_WARN("Received empty mission");
    }
}


void RbpfSlam::synch_cb(const std_msgs::Bool::ConstPtr& finished_msg)
{
    ROS_DEBUG("PF node: Survey finished received");
    mission_finished_ = true;
    save_gp_maps(finished_msg->data);
    ROS_DEBUG("We done bitches, this time in c++");
}   


void RbpfSlam::mbes_real_cb(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    if (mission_finished_ != true && start_training_)
    {
        // Beams in vehicle mbes frame
        // Store in pings history
        // auto t1 = high_resolution_clock::now();
        mbes_history_.emplace_back(Pointcloud2msgToEigen(*msg, beams_real_));

        // Store latest mbes msg for timing
        latest_mbes_ = *msg;

        pings_idx_.push_back(count_pings_);
        count_pings_ += 1;
    }
}

void RbpfSlam::rbpf_update(const ros::TimerEvent&)
{
    if(!mission_finished_)
    {
        if(latest_mbes_.header.stamp > prev_mbes_.header.stamp)
        {
            prev_mbes_ = latest_mbes_;
            if(start_training_){
                this->update_particles_weights(latest_mbes_, odom_latest_);
            }
        }
    }
}

void RbpfSlam::odom_callback(const nav_msgs::OdometryConstPtr& odom_msg)
{
    time_ = odom_msg->header.stamp.toSec();
    odom_latest_ = *odom_msg;

    // Flag to finish the mission
    if(mission_finished_ != true && start_training_)
    {
        // Motion prediction
        if (time_ > old_time_) 
        { 
            this->predict(*odom_msg); 
        }
        // Update stats and visual
        // publish_stats(*odom_msg);
    }
    old_time_ = time_;
}

void RbpfSlam::mb_cb(const slam_msgs::MinibatchTrainingGoalConstPtr& goal)
{
    int pc_id = goal->particle_id;

    // Randomly pick mb_size/beams_per_ping pings 
    int mb_size = goal->mb_size;

    std::mt19937 g(rd_());
    int beams_per_ping = 20;
    Eigen::MatrixXf mb_mat(mb_size, 3);
    slam_msgs::MinibatchTrainingResult result;
    Eigen::Vector3f pos_i;
    Eigen::Matrix3f rot_i;
    int ping_i;
    // If enough beams collected to start minibatch training
    if (count_pings_ > (mb_size / beams_per_ping))
    {
        auto t1 = high_resolution_clock::now();
        // Shuffle indexes of pings collected so far and take the first int(mb_size / beams_per_pings)
        std::shuffle(pings_idx_.begin(), pings_idx_.end(), g);
        for (int i = 0; i < int(mb_size / beams_per_ping); i++)
        {
            ping_i = pings_idx_.at(i);
            // Avoid using the latest ping since some particle might not have updated their pose histories
            // std::cout << "Number of pings " << pings_idx_.size() << std::endl;
            // std::cout << "Current ping " << ping_i << std::endl;
            if(ping_i == pings_idx_.size()-1){
                ping_i = pings_idx_.size() - 3;
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
                    std::shuffle(beams_idx_.begin(), beams_idx_.end(), g);
                    for (int b = 0; b < beams_per_ping; b++)
                    {
                        mb_mat.row(i * beams_per_ping + b) = (rot_i * mbes_history_.at(ping_i).row(beams_idx_.at(b)).transpose()
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
        eigenToPointcloud2msg(mbes_pcloud, mb_mat);

        // Set action as success
        result.success = true;
        result.minibatch = mbes_pcloud;
        as_mb_->setSucceeded(result);

        /* For timing */
        auto t2 = high_resolution_clock::now();
        duration<double, std::milli> ms_double = t2 - t1;
        time_avg_ += ms_double.count();
        count_mb_cbs_++;
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

void RbpfSlam::save_gp_maps(const bool plot)
{
    int pings_t = mbes_history_.size()-1;
    Eigen::MatrixXf mbes_mat(pings_t * beams_real_, 3);
    Eigen::Vector3f pos_i;
    Eigen::Matrix3f rot_i;
    slam_msgs::PlotPosteriorGoal goal;
    int ping_i;
    for(int p=0; p<pc_; p++)
    {
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
                // std::lock_guard<std::mutex> lock(particles_.at(p).pc_mutex_);
                pos_i = particles_.at(p).pos_history_.at(index)->at(ping_i - ancestry_sizes_.at(index));
                rot_i = particles_.at(p).rot_history_.at(index)->at(ping_i - ancestry_sizes_.at(index));
            }

            // TODO: get rid of loop with colwise operations on mbes_history
            for (int b = 0; b < mbes_history_.at(ping_i).rows(); b++)
            {
                mbes_mat.row(ping_i * beams_real_ + b) = (rot_i * mbes_history_.at(ping_i).row(b).transpose() + pos_i).transpose();
            }
        }

        // Parse into pointcloud2
        sensor_msgs::PointCloud2 mbes_pcloud;
        eigenToPointcloud2msg(mbes_pcloud, mbes_mat);

        // Set action as success
        goal.pings = mbes_pcloud;
        goal.plot = plot;
        p_plot_acs_.at(p)->sendGoal(goal);
        // We want this calls to be secuential since plotting is GPU-heavy 
        // and not time critical, so we want the particles to do it one by one 
        bool received = p_plot_acs_.at(p)->waitForResult();        
    }

}

void RbpfSlam::update_particles_weights(sensor_msgs::PointCloud2 &mbes_ping, nav_msgs::Odometry &odom)
{

    // Latest ping depths in map frame
    Eigen::MatrixXf latest_mbes = Pointcloud2msgToEigen(mbes_ping, beams_real_);
    latest_mbes.rowwise() += Eigen::Vector3f(0, 0, m2o_mat_(2, 3) + odom.pose.pose.position.z).transpose();
    latest_mbes_z_ = latest_mbes.col(2);

    ROS_INFO("Updating weights");
    Eigen::MatrixXf ping_mat(beams_real_, 3);
    Eigen::Vector3f pos_i;
    Eigen::Matrix3f rot_i;
    slam_msgs::SamplePosteriorGoal goal;
    for(int p=0; p<pc_; p++){
        // Transform x random beams to particle pose in map frame
        // pos_i = particles_.at(p).pos_history_.back();
        // rot_i = particles_.at(p).rot_history_.back();
        {
            // std::lock_guard<std::mutex> lock(particles_.at(p).pc_mutex_);
            pos_i = particles_.at(p).pos_history_.back()->back();
            rot_i = particles_.at(p).rot_history_.back()->back();
        }

        // TODO: get rid of loop with colwise operations on mbes_history
        for (int b = 0; b < mbes_history_.back().rows(); b++)
        {
            ping_mat.row(b) = (rot_i * mbes_history_.back().row(b).transpose() + pos_i).transpose();
        }

        // Parse into pointcloud2
        sensor_msgs::PointCloud2 mbes_pcloud;
        eigenToPointcloud2msg(mbes_pcloud, ping_mat);

        // Asynch goal
        goal.ping = mbes_pcloud;
        p_sample_acs_.at(p)->sendGoal(goal,
                                      boost::bind(&RbpfSlam::sampleCB, this, _1, _2));
    }

    // Collect all updated weights and call resample()
    int w_time_out = 0;
    while (updated_w_ids_.size() < pc_ && ros::ok() && w_time_out < rbpf_period_ * 100.)
    {
        // ROS_INFO("Updating weights");
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
            // std::cout << particles_.at(i).index_ << " " << particles_.at(i).w_ << std::endl;
            weights.push_back(particles_.at(i).w_);
        }
        this->resample(weights);
    }
    else{
        updated_w_ids_.clear();
        ROS_WARN("Lost weights on the way, skipping resampling");
    }
}

void RbpfSlam::sampleCB(const actionlib::SimpleClientGoalState &state,
                        const slam_msgs::SamplePosteriorResultConstPtr &result)
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

void RbpfSlam::predict(nav_msgs::Odometry odom_t)
{
    float dt = float(time_ - old_time_);
    for(int i = 0; i < pc_; i++) 
    {
        particles_.at(i).motion_prediction(odom_t, dt);
        particles_.at(i).update_pose_history();
        // std::cout << "Particle " << i << " size " << particles_.at(i).pos_history_.back()->size() << std::endl;
    }
}

void RbpfSlam::update_rviz(const ros::TimerEvent &)
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
}

void RbpfSlam::resample(vector<double> weights)
{
    int N_eff = pc_;
    vector<int> indices;
    vector<int> lost;
    vector<int> keep;
    vector<int> dupes;
    std::vector<int>::iterator idx;

    // Normalize weights
    double w_sum = accumulate(weights.begin(), weights.end(), 0);
    if(w_sum == 0)
        ROS_WARN("All weights are zero!");
    else
    {
        transform(weights.begin(), weights.end(), weights.begin(), [&w_sum](auto& c){return c/w_sum;});
        double w_sq_sum = inner_product(begin(weights), end(weights), begin(weights), 0.0); // Sum of squared elements of weigsth
        N_eff = 1 / w_sq_sum;
    }

    n_eff_mask_.erase(n_eff_mask_.begin());
    n_eff_mask_.push_back(N_eff);
    n_eff_filt_ = moving_average(n_eff_mask_, 3);
    std::cout << "Mask " << N_eff << " N_thres " << std::round(pc_ / 2) << std::endl;
    if (N_eff < std::round(pc_ / 2))
    // if(lc_detected_)
    {
        // Resample particles
        ROS_INFO("Resampling");
        indices = systematic_resampling(weights);
        // For manual lc testing
        // indices = vector<int>(pc_, 0);
        // indices = {0, 0, 0, 0, 0, 0, 0, 5, 1, 1, 1, 1, 1, 1, 1, 16, 2, 2, 12, 12};
        lc_detected_ = false;
        
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
        reassign_poses(lost, dupes); 

        // Add noise to particles
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
            // std::lock_guard<std::mutex> lock(particles_.at(dupes[i]).pc_mutex_);
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
    vector<int> cum_sum(N);

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
    vector<float> roll;
    vector<float> pitch;
    vector<float> yaw;

    double roll_i, pitch_i, yaw_i;

    for (int i = 0; i < pose_list.poses.size(); i++)
    {
        x.push_back(pose_list.poses[i].position.x);
        y.push_back(pose_list.poses[i].position.y);
        z.push_back(pose_list.poses[i].position.z);

        // Quaternion to Euler
        tf::Quaternion q(
            pose_list.poses[i].orientation.x,
            pose_list.poses[i].orientation.y,
            pose_list.poses[i].orientation.z,
            pose_list.poses[i].orientation.w);
        tf::Matrix3x3 m(q);
        m.getRPY(roll_i, pitch_i, yaw_i);

        // Wrap yaw around -pi, pi
        yaw_i = fmod(yaw_i + M_PI, 2*M_PI);
        if (yaw_i < 0)
            yaw_i += 2*M_PI;
        yaw_i -= M_PI;

        roll.push_back(roll_i);
        pitch.push_back(pitch_i);
        yaw.push_back(yaw_i);
    }

    // Compute averages
    float x_ave = accumulate(x.begin(), x.end(), 0.0) / x.size();
    float y_ave = accumulate(y.begin(), y.end(), 0.0) / y.size();
    float z_ave = accumulate(z.begin(), z.end(), 0.0) / z.size();
    float roll_ave = accumulate(roll.begin(), roll.end(), 0.0) / roll.size();
    float pitch_ave = accumulate(pitch.begin(), pitch.end(), 0.0) / pitch.size();
    float yaw_ave = accumulate(yaw.begin(), yaw.end(), 0.0) / yaw.size();

    avg_pose_.pose.pose.position.x = x_ave;
    avg_pose_.pose.pose.position.y = y_ave;
    avg_pose_.pose.pose.position.z = z_ave;

    tf2::Quaternion avg_q;
    avg_q.setRPY(roll_ave, pitch_ave, yaw_ave);
    avg_q = avg_q.normalize();
    avg_pose_.pose.pose.orientation.x = avg_q.getX();
    avg_pose_.pose.pose.orientation.y = avg_q.getY();
    avg_pose_.pose.pose.orientation.z = avg_q.getZ();
    avg_pose_.pose.pose.orientation.w = avg_q.getW();
    
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
    stats.data[0]  = n_eff_filt_;
    stats.data[1]  = pc_/2.f;
    stats.data[2]  = gt_odom.pose.pose.position.x;
    stats.data[3]  = gt_odom.pose.pose.position.y;
    stats.data[4]  = gt_odom.pose.pose.position.z;
    stats.data[5]  = avg_pose_.pose.pose.position.x;
    stats.data[6]  = avg_pose_.pose.pose.position.y;
    stats.data[7]  = avg_pose_.pose.pose.position.z;
    stats.data[8]  = cov_(0, 0);
    stats.data[9]  = cov_(0, 1);
    stats.data[10] = cov_(0, 2);
    stats.data[11] = cov_(1, 1);
    stats.data[12] = cov_(1, 2);
    stats.data[13] = cov_(2, 2);

    stats_.publish(stats);
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