#include "sss_particle_filter/sss_pf_localization.h"

pfLocalization::pfLocalization()
{
}

pfLocalization::pfLocalization(ros::NodeHandle &nh, ros::NodeHandle &nh_mb) : 
        nh_(&nh), nh_mb_(&nh_mb), rng_((std::random_device())()), g_((std::random_device())())
{
    // Get parameters from launch file
    nh_->param<int>(("particle_count"), pc_, 10);
    nh_->param<int>(("n_beams_sss"), sss_bin_num_, 512);
    nh_->param<string>(("map_frame"), map_frame_, "map");
    nh_->param<string>(("sss_link"), sss_frame_, "sss_link");
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

    // Timer for end of mission: finish when no more odom is being received
    mission_finished_ = false;
    time_wo_motion_ = 5.;

    // Transforms from auv_2_ros
            
    tf2_ros::TransformListener tf_listener(tf_buffer_);
    try
    {
        ROS_DEBUG("Waiting for transforms");
        auto asynch_1 = std::async(std::launch::async, [this]
                                       { return tf_buffer_.lookupTransform(base_frame_, sss_frame_,
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

        ROS_INFO("Transforms locked - PF node");
    }
    catch (const std::exception &e)
    {
        ROS_ERROR("ERROR: Could not lookup transform from base_link to mbes_link");
    }

    // Subscription to the end of mission topic
    nh_->param<string>(("survey_finished_top"), finished_top_, "/survey_finished");
    finished_sub_ = nh_->subscribe(finished_top_, 100, &pfLocalization::synch_cb, this);
    survey_finished_ = false;

    // Subscription to the save topic
    save_sub_ = nh_->subscribe("/save", 100, &pfLocalization::save_cb, this);

    // Start timing now
    time_ = ros::Time::now().toSec();
    old_time_ = ros::Time::now().toSec();

    // Timer for the RBPF LC prompting
    // nh_mb_->param<float>(("rbpf_period"), rbpf_period_, 0.3);
    // timer_rbpf_ = nh_mb_->createTimer(ros::Duration(rbpf_period_), &pfLocalization::pf_update, this, false);

    // Timer for updating RVIZ
    nh_->param<float>(("rviz_period"), rviz_period_, 0.3);
    if(rviz_period_ != 0.){
        timer_rviz_ = nh_->createTimer(ros::Duration(rviz_period_), &pfLocalization::update_rviz, this, false);
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

    // Initialize the particles on top of vehicle 
    tf::StampedTransform o2b_tf;
    tfListener_.waitForTransform(odom_frame_, base_frame_, ros::Time(0), ros::Duration(300.0));
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
    //     particles_.emplace_back(pfParticle(beams_real_, pc_, i, base2mbes_mat_, m2o_mat_, init_p_pose_,
    //                                        std::vector<float>(6, 0.), meas_std_, motion_cov_));
    // }
    
    // Create particles
    std::string particle_sss_top;
    nh_->param<string>(("particle_sss_top"), particle_sss_top, "/resample_top");
    std::string mesh_resources_path;
    nh.getParam(("mesh_resources"), mesh_resources_path);
    for (int i=0; i<pc_; i++){
        particles_.emplace_back(pfParticle(sss_bin_num_, pc_, i, base2mbes_mat_, m2o_mat_, init_p_pose_,
                                           init_cov_, meas_std_, motion_cov_, mesh_resources_path));

        p_sss_pubs_.push_back(nh_->advertise<auv_model::Sidescan>(particle_sss_top + "/particle_" + std::to_string(i), 1));
    }

    // Dead reckoning particle
    dr_particle_.emplace_back(pfParticle(sss_bin_num_, pc_, pc_, base2mbes_mat_, m2o_mat_, init_p_pose_,
                                         std::vector<float>(6, 0.), meas_std_, std::vector<float>(6, 0.), mesh_resources_path));

    std::string enable_lc_top;
    nh_->param<string>(("particle_enable_lc"), enable_lc_top, "/enable_lc");
    enable_lc_sub_ = nh_->subscribe(enable_lc_top, 100, &pfLocalization::enable_lc, this);

    std::string gps_saved_top;
    nh_->param<string>(("rbpf_saved_top"), gps_saved_top, "/gt/rbpf_saved");
    gp_saved_pub_ = nh_->advertise<std_msgs::Bool>(gps_saved_top, 10);

    // For Hugin markers
    std::string rbpf_markers_top;
    nh_->param<string>(("markers_top"), rbpf_markers_top, "/markers");
    vis_pub_ = nh_->advertise<visualization_msgs::MarkerArray>(rbpf_markers_top, 0);

    nh_->param<string>(("expected_sss_as"), sss_sim_as_, "/dr_pose");
    ac_sss_ = new actionlib::SimpleActionClient<auv_model::SssSimAction>(sss_sim_as_, true);

    while (!ac_sss_->waitForServer(ros::Duration(1.0)) && ros::ok())
    {
        ROS_INFO_NAMED("PF ", "SSS waiting for action server");
    }

    // Subscription to manually triggering LC detection. Just for testing
    nh_->param<string>(("lc_manual_topic"), lc_manual_topic_, "/manual_lc");
    lc_manual_sub_ = nh_->subscribe(lc_manual_topic_, 1, &pfLocalization::manual_lc, this);
    
    // Aux variables for submaps
    ping_cnt_ = 0;
    submap_size_ = 100;
    ancestry_sizes_.push_back(0);

    // Subscription to sss pings
    sss_full_image_ = cv::Mat::zeros(1, sss_bin_num_ * 2, CV_8UC1);
    nh_->param<string>(("sss_pings_topic"), sss_pings_top_, "sss_pings");
    sss_sub_ = nh_mb_->subscribe(sss_pings_top_, 10, &pfLocalization::sss_cb, this);

    // Establish subscription to odometry message (intentionally last)
    nh_->param<string>(("odometry_topic"), odom_top_, "odom");
    odom_sub_ = nh_->subscribe(odom_top_, 100, &pfLocalization::odom_callback, this);

    ROS_INFO("Particle filter instantiated");
}

bool pfLocalization::empty_srv(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
{
    ROS_DEBUG("PF ready");
    return true;
}

void pfLocalization::manual_lc(const std_msgs::Bool::ConstPtr& lc_msg) { lc_detected_ = true; }

// Receive IDs of SVGPs when they are ready to be used for LC prompting 
void pfLocalization::enable_lc(const std_msgs::Int32::ConstPtr& enable_lc)
{
    svgp_lc_ready_.push_back(enable_lc->data);
}

void pfLocalization::synch_cb(const std_msgs::Bool::ConstPtr& finished_msg)
{
    ROS_INFO("RBPF node: Survey finished received");
    mission_finished_ = true;
    // save_gps(false);
    ROS_INFO("We done bitches");
}


void pfLocalization::save_cb(const std_msgs::Bool::ConstPtr& save_msg)
{
    ROS_INFO("PF node: saving without finishing the mission");
    // save_gps(save_msg->data);
    ROS_INFO("Aaaand it's saved");
}

void pfLocalization::sss_cb(const auv_model::SidescanConstPtr &msg)
{
    if (mission_finished_ != true)
    {
        // Beams in vehicle mbes frame
        // Store in pings history
        // std::cout << "Starting sss cb " << ping_cnt_ << std::endl;
        auto t1 = high_resolution_clock::now();

        cv::Mat port(1, msg->port_channel.size(), CV_8UC1);
        cv::Mat stbd(1, msg->starboard_channel.size(), CV_8UC1);
        for (size_t i = 0; i < msg->port_channel.size(); ++i)
        {
            port.at<uint8_t>(0, i) = msg->port_channel[i];
            stbd.at<uint8_t>(0, i) = msg->starboard_channel[i];
        }

        // Flip the port channel and concatenate and store
        cv::Mat flippedPort;
        cv::flip(port, flippedPort, 1); // Flip horizontally (axis = 1)
        cv::Mat meas;
        cv::hconcat(flippedPort, stbd, meas);
        sss_full_image_.push_back(meas);
        ping_cnt_ ++;

        // Store in history particles poses corresponding to the current ping
        this->update_particles_history();

        // Compute expected measurements for every particle in their latest pose
        // this->expected_measurements(odom_latest_);
        // this->meas_predict_particles();

        if (ping_cnt_ % submap_size_ == 0)
        {
            std::cout << "Saving real patch image " << sss_full_image_.rowRange(sss_full_image_.rows - submap_size_, sss_full_image_.rows).rows << std::endl;
            this->compute_weights(sss_full_image_.rowRange(sss_full_image_.rows - submap_size_, sss_full_image_.rows));

            std::string filename = "./sss_real_patch_"+ std::to_string(ping_cnt_) + ".png";
            bool success = cv::imwrite(filename, sss_full_image_.rowRange(sss_full_image_.rows - submap_size_, sss_full_image_.rows));

            std::vector<double> weights;
            for (int i = 0; i < pc_; i++)
            {
                std::cout << "Particle " << i << ", weight " << particles_.at(i).w_ << std::endl;
                weights.push_back(particles_.at(i).w_);
            }
            this->resample(weights);
        }

        // std::cout << "Done with sss cb" << std::endl;
        // auto t2 = high_resolution_clock::now();
        // duration<double, std::milli> ms_double = t2 - t1;
        // std::cout << ms_double.count() / 1000.0 << std::endl;
        // count_mbes_cbs_++;
        // time_avg_ += ms_double.count();
        // std::cout << (time_avg_ / double(count_mb_cbs_))/1000.0 << std::endl;
        // std::cout << count_mb_cbs_ << std::endl;
    }
}

void pfLocalization::odom_callback(const nav_msgs::OdometryConstPtr& odom_msg)
{
    time_ = odom_msg->header.stamp.toSec();
    odom_latest_ = *odom_msg;

    // // Flag to finish the mission
    // if(mission_finished_ != true && start_training_)
    if(mission_finished_ != true)
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


void pfLocalization::expected_measurements(nav_msgs::Odometry &odom)
{
    Eigen::Vector3f pos_i;
    Eigen::Matrix3f rot_i;
    slam_msgs::ManipulatePosteriorGoal goal;
    for (int p = 0; p < pc_; p++)
    {
        {
            // Latest particle pose
            std::lock_guard<std::mutex> lock(*particles_.at(p).pc_mutex_);
            pos_i = particles_.at(p).pos_history_.back()->back();
            rot_i = particles_.at(p).rot_history_.back()->back();
        }

        // TODO: this is sequential, change to asynch eventually
        
        // Rebuilt particle pose as geometry msg
        Eigen::Matrix4f pose_i_mat = Eigen::Matrix4f::Identity();
        pose_i_mat.topLeftCorner(3, 3) = rot_i.matrix();
        pose_i_mat.block(0, 3, 3, 1) = pos_i.head(3);
        Eigen::Affine3d sss_pose_eigen;
        sss_pose_eigen.matrix() = pose_i_mat.cast<double>();
        geometry_msgs::Transform sss_pose_msg;
        tf::transformEigenToMsg(sss_pose_eigen, sss_pose_msg);

        auv_model::SssSimGoal sss_goal;
        sss_goal.sss_pose.header.frame_id = map_frame_;
        sss_goal.sss_pose.child_frame_id = sss_frame_;
        sss_goal.sss_pose.header.stamp = odom.header.stamp;
        sss_goal.sss_pose.transform = sss_pose_msg;
        sss_goal.beams_num.data = sss_bin_num_;
        ac_sss_->sendGoal(sss_goal);

        ac_sss_->waitForResult(ros::Duration(1.0));
        actionlib::SimpleClientGoalState state = ac_sss_->getState();

        // If expected SSS meas received, store in the particle's submap
        if (state == actionlib::SimpleClientGoalState::SUCCEEDED)
        {
            auv_model::SssSimResult sss_res = *ac_sss_->getResult();

            cv::Mat port(1, sss_res.sim_sss.port_channel.size(), CV_8UC1);
            cv::Mat stbd(1, sss_res.sim_sss.starboard_channel.size(), CV_8UC1);
            for (size_t i = 0; i < sss_res.sim_sss.port_channel.size(); ++i)
            {
                port.at<uint8_t>(0, i) = sss_res.sim_sss.port_channel[i];
                stbd.at<uint8_t>(0, i) = sss_res.sim_sss.starboard_channel[i];
            }

            // Flip the port channel and concatenate and store
            cv::Mat flippedPort;
            cv::flip(port, flippedPort, 1); // Flip horizontally (axis = 1)
            cv::Mat meas;
            cv::hconcat(flippedPort, stbd, meas);
            particles_.at(p).sss_patch_.push_back(meas);

            // Republish for debugging
            auv_model::Sidescan sss_msg;
            sss_msg = sss_res.sim_sss;
            p_sss_pubs_.at(p).publish(sss_msg);
        }
    }
}

void pfLocalization::compute_weights(const cv::Mat real_sss_patch)
{
    auto t1 = high_resolution_clock::now();

    // Multithreading
    for (int i = 0; i < pc_; i++)
    {
        // Passing real_sss_patch by reference yields issues. Create copies instead
        weights_threads_vec_.emplace_back(std::thread(&pfParticle::compute_weight_sss,
                                                    &particles_.at(i), real_sss_patch));
        // Sequential version   
        // particles_.at(i).compute_weight_sss(real_sss_patch);
    }

    for (int i = 0; i < pc_; i++)
    {
        if (weights_threads_vec_[i].joinable())
        {
            weights_threads_vec_[i].join();
        }
    }
    weights_threads_vec_.clear();
    // ROS_INFO("Threads weights done");
}

// void pfLocalization::meas_predict_particles()
// {
//     // Multithreading
//     auto t1 = high_resolution_clock::now();
//     for (int i = 0; i < pc_; i++)
//     {
//         // particles_.at(i).sss_prediction();
//         meas_threads_vec_.emplace_back(std::thread(&pfParticle::sss_prediction, &particles_.at(i)));
//     }
//     // ROS_INFO("Threads meas launched");

//     for (int i = 0; i < pc_; i++)
//     {
//         if (meas_threads_vec_[i].joinable())
//         {
//             meas_threads_vec_[i].join();
//         }
//         // p_sss_pubs_.at(i).publish(particles_.at(i).sss_msg_);
//     }
//     meas_threads_vec_.clear();

//     auto t2 = high_resolution_clock::now();
//     avg_time.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
//     if (avg_time.size() > 10.)
//         avg_time.erase(avg_time.begin());
        
//     std::cout << "Ping projection time (ms) " << std::accumulate(avg_time.begin(), avg_time.end(), 0) / 10. << std::endl;
// }

void pfLocalization::predict(nav_msgs::Odometry odom_t, float dt)
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
        pred_threads_vec_.emplace_back(std::thread(&pfParticle::motion_prediction, 
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
    // ROS_INFO("Threads motion prediction done");

    // auto t2 = high_resolution_clock::now();
    // duration<double, std::milli> ms_double = t2 - t1;
    // std::cout << ms_double.count() / 1000.0 << std::endl;
}

void pfLocalization::update_particles_history()
{
    auto t1 = high_resolution_clock::now();

    for(int i = 0; i < pc_; i++)
    {
        upd_threads_vec_.emplace_back(std::thread(&pfParticle::update_pose_history, 
                                    &particles_.at(i)));
    }

    for (int i = 0; i < pc_; i++)
    {
        if (upd_threads_vec_[i].joinable())
        {
            upd_threads_vec_[i].join();
        }
        p_sss_pubs_.at(i).publish(particles_.at(i).sss_msg_);
    }
    upd_threads_vec_.clear();

    auto t2 = high_resolution_clock::now();
    avg_time.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count());
    if (avg_time.size() > 10.)
        avg_time.erase(avg_time.begin());

    std::cout << "Ping projection time (ms) " << std::accumulate(avg_time.begin(), avg_time.end(), 0) / 10. << std::endl;
}

void pfLocalization::resample(vector<double> weights)
{
    N_eff_ = pc_;
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
    if (w_sum == 0.)
        ROS_WARN("All weights are zero!");
    else
    {
        // Normalize weights
        transform(weights.begin(), weights.end(), weights.begin(), [&w_sum](auto &c)
                  { return c / w_sum; });
        std::cout << "Normalized weights :" << std::endl;
        for (auto weight : weights)
        {
            std::cout << weight << " ";
        }
        std::cout << std::endl;

        // Compute effective number
        double w_sq_sum = inner_product(begin(weights), end(weights), begin(weights), 0.0); // Sum of squared elements of weigsth
        N_eff_ = 1 / w_sq_sum;
    }

    // n_eff_mask_.erase(n_eff_mask_.begin());
    // n_eff_mask_.push_back(N_eff);
    // n_eff_filt_ = moving_average(n_eff_mask_, 3);

    std::cout << std::endl;
    std::cout << "Mask exp " << N_eff_ << " N_thres " << std::round(pc_ / 2) << std::endl;

    if (N_eff_ < std::round(pc_ / 2) || lc_detected_)
    // if (N_eff < 90 || lc_detected_)
    {
        // Resample particles
        ROS_INFO("Resampling");
        indices = systematic_resampling(weights);

        // For manual lc testing
        if (lc_detected_)
        {
            indices = vector<int>(pc_, 0);
        }

        set<int> s(indices.begin(), indices.end());
        keep.assign(s.begin(), s.end());
        for (int i = 0; i < pc_; i++)
        {
            if (!count(keep.begin(), keep.end(), i))
                lost.push_back(i);
        }

        dupes = indices;
        std::cout << "Keep " << std::endl;
        for (int i : keep)
        {
            std::cout << i << " ";
            if (count(dupes.begin(), dupes.end(), i))
            {
                idx = find(dupes.begin(), dupes.end(), i);
                dupes.erase(idx);
            }
        }
        std::cout << std::endl;

        // Reasign and ddd noise to particles poses
        ROS_INFO("Reasigning poses");
        reassign_poses(lost, dupes);
        for (int i = 0; i < pc_; i++)
            particles_[i].add_noise(res_noise_cov_);

        lc_detected_ = false;

        // auto t2 = high_resolution_clock::now();
        // duration<double, std::milli> ms_double = t2 - t1;
        // std::cout << ms_double.count() / 1000.0 << std::endl;
    }
}

void pfLocalization::pub_markers(const geometry_msgs::PoseArray& array_msg)
{
    visualization_msgs::MarkerArray markers;
    int i = 0;
    for (geometry_msgs::Pose pose_i : array_msg.poses)
    {
        visualization_msgs::Marker marker;
        marker.header.frame_id = odom_frame_;
        marker.header.stamp = ros::Time();
        marker.ns = "markers";
        marker.id = i;
        marker.type = visualization_msgs::Marker::MESH_RESOURCE;
        marker.action = visualization_msgs::Marker::ADD;
        marker.pose.position.x = pose_i.position.x;
        marker.pose.position.y = pose_i.position.y;
        marker.pose.position.z = pose_i.position.z;
        marker.pose.orientation.x = pose_i.orientation.x;
        marker.pose.orientation.y = pose_i.orientation.y;
        marker.pose.orientation.z = pose_i.orientation.z;
        marker.pose.orientation.w = pose_i.orientation.w;
        marker.scale.x = 0.001;
        marker.scale.y = 0.001;
        marker.scale.z = 0.001;
        marker.color.a = 1.0; 
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.mesh_resource = "package://hugin_description/mesh/Hugin_big_meter.dae";
        markers.markers.push_back(marker);
        i++;
    }

    vis_pub_.publish(markers);
}

void pfLocalization::update_rviz(const ros::TimerEvent &)
{
    // if(start_training_)
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
        // TODO: fix
        // average_pose(array_msg);
        
        if(false){
            pf_pub_.publish(array_msg);
        }
        else{
            pub_markers(array_msg);
        }

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

void pfLocalization::reassign_poses(vector<int> lost, vector<int> dupes)
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

vector<int> pfLocalization::systematic_resampling(vector<double> weights)
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

void pfLocalization::average_pose(geometry_msgs::PoseArray pose_list)
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

void pfLocalization::publish_stats(nav_msgs::Odometry gt_odom)
{
    // Send statistics for visualization
    std_msgs::Float32MultiArray stats;
    stats.data.push_back(float(N_eff_));
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

float pfLocalization::moving_average(vector<int> a, int n)
{
    vector<float> a_last(a.end() - n, a.end());
    float a_sum = accumulate(a_last.begin(), a_last.end(), 0);

    return (float)a_sum/n;
}

std::vector<int> pfLocalization::arange(int start, int stop, int step)
{
    std::vector<int> values;
    for (int value = start; value < stop; value += step)
        values.push_back(value);
    return values;
}

pfLocalization::~pfLocalization()
{
    delete (nh_);
    delete (nh_mb_);
}
