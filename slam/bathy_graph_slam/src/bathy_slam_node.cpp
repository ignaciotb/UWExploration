#include <bathy_graph_slam/bathy_slam.hpp>


BathySlamNode::BathySlamNode(std::string node_name, ros::NodeHandle &nh): node_name_(node_name), nh_(&nh)
{

    std::string pings_top, debug_pings_top, odom_top, sim_pings_top, enable_top, synch_top;
    nh_->param<std::string>("mbes_pings", pings_top, "/gt/mbes_pings");
    nh_->param<std::string>("odom_gt", odom_top, "/gt/odom");
    nh_->param<std::string>("map_frame", map_frame_, "map");
    nh_->param<std::string>("odom_frame", odom_frame_, "odom");
    nh_->param<std::string>("base_link", base_frame_, "base_frame");
    nh_->param<std::string>("mbes_link", mbes_frame_, "mbes_frame");
    nh_->param<std::string>("survey_finished_top", enable_top, "enable");
    nh_->param<std::string>("synch_topic", synch_top, "slam_synch");

    // Synchronizer for MBES and odom msgs
    mbes_subs_.subscribe(*nh_, pings_top, 1);
    odom_subs_.subscribe(*nh_, odom_top, 1);
    synch_ = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), mbes_subs_, odom_subs_);
    synch_->registerCallback(&BathySlamNode::pingCB, this);

    submaps_pub_ = nh_->advertise<sensor_msgs::PointCloud2>("/submaps", 10, false);
    enable_subs_ = nh_->subscribe(enable_top, 1, &BathySlamNode::enableCB, this);

    try {
        tflistener_.waitForTransform(mbes_frame_, base_frame_, ros::Time(0), ros::Duration(10.0) );
        tflistener_.lookupTransform(mbes_frame_, base_frame_, ros::Time(0), tf_mbes_base_);
        ROS_INFO("Locked transform base --> sensor");
    }
    catch(tf::TransformException &exception) {
        ROS_ERROR("%s", exception.what());
        ros::Duration(1.0).sleep();
    }

    // Initialize survey params
    first_msg_ = true;
    submaps_cnt_ = 0;

    // ISAM solver
    isam_obj = new samGraph();

    // Empty service to synch the applications waiting for this node to start
    synch_service_ = nh_->advertiseService(synch_top,
                                            &BathySlamNode::emptySrv, this);
    ROS_INFO("Initialized SLAM");
}

bool BathySlamNode::emptySrv(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res)
{
}

void BathySlamNode::updateTf()
{
    int cnt_i = 0;
    tf::StampedTransform tf_map_submap_stp;
    geometry_msgs::TransformStamped msg_map_submap;
    for (tf::Transform &tf_measi_map : tf_submaps_vec_)
    {
        tf_map_submap_stp = tf::StampedTransform(tf_measi_map,
                                                 ros::Time::now(),
                                                 odom_frame_,
                                                 "submap_" + std::to_string(cnt_i));

        cnt_i += 1;
        tf::transformStampedTFToMsg(tf_map_submap_stp, msg_map_submap);
        submaps_bc_.sendTransform(msg_map_submap);
    }

    // For RVIZ
    if (submaps_vec_.empty())
    {
        std::cout << "Submaps vec empty " << std::endl;
    }
    sensor_msgs::PointCloud2 submap_msg;
    pcl::toROSMsg(submaps_vec_.at(submaps_cnt_ - 1).submap_pcl_, submap_msg);
    submap_msg.header.frame_id = "submap_" + std::to_string(tf_submaps_vec_.size() - 1);
    submap_msg.header.stamp = ros::Time::now();
    submaps_pub_.publish(submap_msg);
    std::cout << "Submaps constructed" << std::endl;
}

void BathySlamNode::pingCB(const sensor_msgs::PointCloud2Ptr &mbes_ping, const nav_msgs::OdometryPtr &odom_msg)
{
    // Set prior on first odom pose
    if (first_msg_){
        isam_obj->addPrior();
        first_msg_ = false;
    }

    tf::Transform ping_tf;
    tf::poseMsgToTF(odom_msg->pose.pose, ping_tf);
    submap_raw_.emplace_back(mbes_ping, ping_tf * tf_mbes_base_);

    if (submap_raw_.size() > 20)
    {
        this->addSubmap();
        submap_raw_.clear();
    }
}


// Callback to finish mission
void BathySlamNode::enableCB(const std_msgs::BoolPtr &enable_msg)
{
    if(enable_msg->data == true){
        ROS_INFO_STREAM("Creating benchmark");
        benchmark::track_error_benchmark benchmark("real_data");
        PointsT gt_map = pclToMatrixSubmap(submaps_vec_);
        PointsT gt_track = trackToMatrixSubmap(submaps_vec_);
        ROS_INFO_STREAM("About to... " << gt_track.size());
        benchmark.add_ground_truth(gt_map, gt_track);
    }
}

Pose2 BathySlamNode::odomStep(unsigned int odom_step)
{
    // // Rotation
    // tf::Quaternion q_prev, q_now;
    // tf::quaternionMsgToTF(submaps_vec_.at(submaps_cnt_ - 2).submap_tf_.rotation, q_prev);
    // tf::quaternionMsgToTF(submaps_vec_.at(submaps_cnt_ - 1).submap_tf_.rotation, q_now);

    // tf::Quaternion q_step = q_now.normalized() * q_prev.inverse().normalized();
    // tf::Matrix3x3 m_step(q_step);
    // double roll_step, pitch_step, yaw_step;
    // m_step.getRPY(roll_step, pitch_step, yaw_step);

    // // Translation
    // Eigen::Vector3d pos_now, pos_prev;
    // tf::vectorMsgToEigen(submaps_vec_.at(submaps_cnt_ - 2).submap_tf_.translation, pos_prev);
    // tf::vectorMsgToEigen(submaps_vec_.at(submaps_cnt_ - 1).submap_tf_.translation, pos_now);
    // Eigen::Vector3d pos_step = pos_now - pos_prev;

    // return Pose2(pos_step[0], pos_step[1], yaw_step);
}

void BathySlamNode::addSubmap()
{
    std::cout << "Calling submap constructor" << std::endl;

    // Store submap tf
    tf::Transform tf_submap_i = std::get<1>(submap_raw_.at(submap_raw_.size()/2));
    tf_submaps_vec_.push_back(tf_submap_i);

    // Create submap object
    SubmapObj submap_i;
    Eigen::Affine3d tf_affine;
    tf::transformTFToEigen(tf_submap_i, tf_affine);
    submap_i.submap_tf_ = tf_affine.matrix().cast<float>();

    for(ping_raw& ping_j: submap_raw_){
        PointCloudT pcl_ping;
        pcl::fromROSMsg(*std::get<0>(ping_j).get(), pcl_ping);
        pcl_ros::transformPointCloud(pcl_ping, pcl_ping, tf_submap_i.inverse() * std::get<1>(ping_j));
        submap_i.submap_pcl_ += pcl_ping;
    }

    submap_i.submap_pcl_.sensor_origin_ << submap_i.submap_tf_.translation();
    submap_i.submap_pcl_.sensor_orientation_ = submap_i.submap_tf_.linear();
    submap_i.submap_id_ = submaps_cnt_;

    submap_i.auv_tracks_ = submap_i.submap_tf_.translation().transpose().cast<double>();

    submaps_cnt_++;
    submaps_vec_.push_back(submap_i);
    
    // Update graph DR
    // Pose2 odom_step = this->odomStep(submaps_cnt_);
    // odom_step.print("Latest odom step ");
    // isam_obj->addOdomFactor();


    // Update TF
    this->updateTf();
}



int main(int argc, char** argv){

    ros::init(argc, argv, "bathy_slam_node");
    ros::NodeHandle nh("~");

    BathySlamNode* bathy_slam = new BathySlamNode(ros::this_node::getName(), nh);

    // ros::Timer timer = nh.createTimer(ros::Duration(1), &BathySlamNode::bcMapSubmapsTF, bathy_slam);

    ros::spin();
    ros::waitForShutdown();

    if(!ros::ok()){
        delete bathy_slam;
    }
    ROS_INFO("Bathy SLAM node finished");

    return 0;
}
