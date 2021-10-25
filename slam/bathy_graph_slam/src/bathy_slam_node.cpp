#include <bathy_graph_slam/bathy_slam.hpp>

BathySlamNode::BathySlamNode(std::string node_name, ros::NodeHandle &nh) : node_name_(node_name),
                                                                           nh_(&nh)
{
    std::string debug_pings_top, odom_top, synch_top, submap_top, indexes_top;
    nh_->param<std::string>("odom_topic", odom_top, "/gt/odom");
    nh_->param<std::string>("map_frame", map_frame_, "map");
    nh_->param<std::string>("odom_frame", odom_frame_, "odom");
    nh_->param<std::string>("base_link", base_frame_, "base_frame");
    nh_->param<std::string>("mbes_link", mbes_frame_, "mbes_frame");
    nh_->param<std::string>("synch_topic", synch_top, "slam_synch");
    nh_->param<std::string>("submaps_topic", submap_top, "/submaps");
    nh_->param<std::string>("landmarks_idx_topic", indexes_top, "/lm_idx");
    nh_->param<std::string>("graph_init_path", graph_init_path_, "./graph_solved.txt");
    nh_->param<std::string>("graph_solved_path", graph_solved_path_, "./graph_solved.txt");

    // submap_subs_ = nh_->subscribe(submap_top, 3, &BathySlamNode::updateGraphCB, this);
    odom_subs_ = nh_->subscribe(odom_top, 30, &BathySlamNode::odomCB, this);

    // Synchronizer for landmarks point cloud and its vector of indexes
    lm_subs_.subscribe(*nh_, submap_top, 30);
    lm_idx_subs_.subscribe(*nh_, indexes_top, 30);
    synch_ = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(30), 
                                                             lm_subs_, lm_idx_subs_);
    synch_->registerCallback(&BathySlamNode::updateGraphCB, this);

    try
    {
        tflistener_.waitForTransform(map_frame_, odom_frame_, ros::Time(0), ros::Duration(20.0));
        tflistener_.lookupTransform(map_frame_, odom_frame_, ros::Time(0), tf_map_odom_);
        ROS_INFO_NAMED(node_name_, " locked transform map --> odom");
    }
    catch (tf::TransformException &exception)
    {
        ROS_ERROR("%s", exception.what());
        ros::Duration(1.0).sleep();
    }

    // Initialize survey params
    first_msg_ = true;
    submaps_cnt_ = 1;
    odomVec_.clear();
    // Initial pose is on top of odom frame
    prev_submap_odom_.reset(new nav_msgs::Odometry());
    prev_submap_odom_->header.frame_id = odom_frame_;
    prev_submap_odom_->child_frame_id = mbes_frame_;
    prev_submap_odom_->header.stamp = ros::Time::now();
    prev_submap_odom_->pose.pose.position.x = 0.0;
    prev_submap_odom_->pose.pose.position.y = 0.0;
    prev_submap_odom_->pose.pose.position.z = 0.0;
    prev_submap_odom_->pose.pose.orientation.x = 0.0;
    prev_submap_odom_->pose.pose.orientation.y = 0.0;
    prev_submap_odom_->pose.pose.orientation.z = 0.0;
    prev_submap_odom_->pose.pose.orientation.w = 1.0;

    // ISAM solver
    graph_solver.reset(new samGraph());

    // Empty service to synch the applications waiting for this node to start
    synch_service_ = nh_->advertiseService(synch_top,
                                            &BathySlamNode::emptySrv, this);
    ROS_INFO_NAMED(node_name_, " initialized");
}

void BathySlamNode::odomCB(const nav_msgs::OdometryPtr &odom_msg)
{
    odomVec_.push_back(*odom_msg);
}

bool BathySlamNode::emptySrv(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res)
{
    return true;
}

void BathySlamNode::updateGraphCB(const sensor_msgs::PointCloud2Ptr &lm_pcl_msg, 
                                  const bathy_graph_slam::LandmarksIdxPtr &lm_idx)
{
    std::cout << "Updating graph " << std::endl;
    // Set prior on first odom pose
    if (first_msg_)
    {
        Eigen::Affine3d map_2_odom;
        tf::transformTFToEigen(tf_map_odom_, map_2_odom);

        Eigen::Quaterniond q_init(map_2_odom.linear());
        Eigen::Vector3d euler_init = q_init.toRotationMatrix().eulerAngles(0, 1, 2);
        Eigen::Vector3d pos_init;
        pos_init = map_2_odom.translation();

        Pose3 poseInit = Pose3(Rot3::Ypr(euler_init[2], euler_init[1], euler_init[0]), 
                                     Point3(pos_init[0], pos_init[1], pos_init[2]));

        graph_solver->addPrior(poseInit);
        first_msg_ = false;
    }

    // Update graph DR concatenating odom msgs between submaps
    Pose3 current_pose;
    Pose3 odom_step = this->odomStep(lm_pcl_msg->header.stamp.toSec(), current_pose);
    graph_solver->addOdomFactor(current_pose, odom_step, submaps_cnt_);

    // Add landmarks to graph as 3D range-bearing measurements from the current DR pose
    std::vector<int> lm_idx_vec = lm_idx->idx.data;
    PointCloudT lm_pcl;
    pcl::fromROSMsg(*lm_pcl_msg, lm_pcl);
    bool lc_detected = graph_solver->addLandmarksFactor(lm_pcl, submaps_cnt_, 
                                                        lm_idx_vec, current_pose);

    // If LCs detected
    if (lc_detected)
    {
        ROS_INFO("Loop closure detected");
        // For testing, save init graph and estimate in disk
        graph_solver->saveResults(*graph_solver->initValues_, graph_init_path_);
        graph_solver->saveSerial(*graph_solver->graph_, * graph_solver->initValues_, graph_init_path_);
        // std::cout << "Computing marginals" << std::endl;
        // Marginals *marginals = new Marginals(*graph_solver->graph_, *graph_solver->initValues_);
        // marginals->print();

        // // Update and solve ISAM2
        // int updateIterations = 10;
        // graph_solver->updateISAM2(updateIterations);
        // Values current_estimate = graph_solver->computeEstimate();
        
        // graph_solver->saveResults(current_estimate, graph_solved_path_);
    }

    submaps_cnt_++;
}

Pose3 BathySlamNode::odomStep(double t_step, Pose3 &current_pose)
{
    // Find odom msg corresponding to submap at time t_step
    // TODO: clear odomVec_ once in a while
    auto current_odom = odomVec_.begin();
    current_odom = std::find_if(current_odom, odomVec_.end(), [&](const nav_msgs::Odometry& odom) {
        return std::abs(odom.header.stamp.toSec() - t_step) < 0.01;
    });
    if(current_odom == odomVec_.end()){
        std::cout << "Not found current odom that matches submap" << std::endl;
    }

    // auto last_odom = odomVec_.begin();
    // last_odom = std::find_if(last_odom, odomVec_.end(), [&](const nav_msgs::Odometry &odom)
    //                          { return std::abs(odom.header.stamp.toSec() - prev_submap_odom_->header.stamp.toSec()) < 0.01; });
    // if (last_odom == odomVec_.end())
    // {
    //     last_odom = odomVec_.begin();
    //     std::cout << "Not found last odom that matches submap" << std::endl;
    // }

    // double step_odom = 0;
    // double t = (odomVec_.begin() + 30)->header.stamp.toSec() - (odomVec_.begin()+29)->header.stamp.toSec();
    // std::cout << "Time step " << t << std::endl;
    // for (auto it = last_odom; it != current_odom; ++it)
    // {
    //     step_odom += it->twist.twist.linear.x * t;
    // }

    // Base pose in odom frame for t and t-1
    Eigen::Affine3d pose_now, pose_prev, map_2_odom;
    tf::transformTFToEigen(tf_map_odom_, map_2_odom);
    tf::poseMsgToEigen (current_odom->pose.pose, pose_now);
    tf::poseMsgToEigen (prev_submap_odom_->pose.pose, pose_prev);

    // Base pose in map frame for t and t-1
    // std::cout << "Pose now " << pose_now.translation().transpose() << std::endl;
    // std::cout << "Pose prev " << pose_prev.translation().transpose() << std::endl;
    Eigen::Affine3d pose_now_map = map_2_odom * pose_now;
    // Eigen::Affine3d pose_now_map = pose_now;
    Eigen::Affine3d pose_prev_map = map_2_odom * pose_prev;
    // Eigen::Affine3d pose_prev_map = pose_prev;

    // Odom step as diff between odom_msgs from two consecutive submaps
    // Rotation
    Eigen::Quaterniond q_prev(pose_prev_map.linear());
    Eigen::Quaterniond q_now(pose_now_map.linear()); 
    Eigen::Quaterniond q_step = q_now.normalized() * q_prev.inverse().normalized();
    Eigen::Vector3d euler_step = q_step.toRotationMatrix().eulerAngles(0, 1, 2);

    // Translation
    Eigen::Vector3d pos_now, pos_prev;
    pos_prev = pose_prev_map.translation();
    pos_now = pose_now_map.translation();
    Eigen::Vector3d pos_step = pos_now - pos_prev;
    double step = pos_step.norm(); //* cos(euler_step[2]);
    // double step = std::sqrt((pos_now ^ 2) + (pos_prev ^ 2));

    // Store odom from current submap as prev
    prev_submap_odom_.reset(new nav_msgs::Odometry(*current_odom));

    // Convert current_odom to gtsam pose for landmarks computation
    Eigen::Vector3d euler_now = q_now.normalized().toRotationMatrix().eulerAngles(0, 1, 2);
    current_pose = Pose3(Rot3::Ypr(euler_now[2], euler_now[1], euler_now[0]), 
                         Point3(pos_now[0], pos_now[1], pos_now[2]));

    std::cout << "Odom step " << step << std::endl;
    // std::cout << "Second Odom step " << step_odom << std::endl;
    std::cout << "Euler angles " << euler_step.transpose() << std::endl;
    // Return odom step as gtsam pose
    return Pose3(Rot3::Ypr(euler_step[2], euler_step[1], euler_step[0]),
                 Point3(step, 0, 0));
}

int main(int argc, char** argv){

    ros::init(argc, argv, "bathy_slam_node");
    ros::NodeHandle nh("~");

    boost::shared_ptr<BathySlamNode> bathySlam;
    bathySlam.reset(new BathySlamNode("bathy_slam_node", nh));

    // ros::Timer timer = nh.createTimer(ros::Duration(1), &BathySlamNode::bcMapSubmapsTF, bathySlam);

    ros::spin();

    ROS_INFO_NAMED(ros::this_node::getName(), " finished");

    return 0;
}
