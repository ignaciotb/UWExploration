#include <bathy_graph_slam/bathy_slam.hpp>

samGraph::samGraph()
{
    // ISAM solver params
    ISAM2Params params = ISAM2Params(ISAM2GaussNewtonParams(0.001), 0.0,
                                     0, false, true,
                                     ISAM2Params::CHOLESKY, true,
                                     DefaultKeyFormatter, true);

    num_landmarks_ = 0;

    // TODO: this has to be an input param
    odoNoise_ = noiseModel::Diagonal::Sigmas((Vector(6) << 0.1, 0.1, 0,1,
                                            0, 0, M_PI / 100.0).finished());
    brNoise_ = noiseModel::Diagonal::Sigmas((Vector(3)<<0.01,0.03,0.05).finished());
    // Instantiate pointers
    isam_.reset(new ISAM2(params));
    graph_.reset(new NonlinearFactorGraph());
    initValues_.reset(new Values());
}

samGraph::~samGraph()
{
}

void samGraph::addPrior(Pose3& initPose)
{
    // Add a prior at time 0 and update isam
    graph_->addPrior(Symbol('x', 0), initPose, odoNoise_);
    initValues_->insert(Symbol('x', 0), initPose);
    // Init last pose where the odom frame is
    lastPose_ = initPose;

    // TODO: do we need to update isam here?
    // isam_->update(*graph_, *initValues_);
    // graph_.reset(new NonlinearFactorGraph());
    // initValues_.reset(new Values());
    ROS_INFO("Prior updated");
}

void samGraph::addOdomFactor(Pose3 factor_pose, Pose3 odom_step, size_t step)
{
    // Add odometry
    // submap i will be DR factor i+1 since the origin 
    // (where there's no submap) is the factor 0
    graph_->emplace_shared<BetweenFactor<Pose3> >(Symbol('x', step), Symbol('x', step+1),
                                                  odom_step, odoNoise_);

    // Predict pose and add as initial estimate
    // Pose3 predictedPose = lastPose_.compose(odom_step);
    // lastPose_ = predictedPose;
    // predictedPose.print("Node added ");
    initValues_->insert(Symbol('x', step+1), factor_pose);
    ROS_INFO("Odom factor added");
}

void samGraph::addLandmarksFactor(PointCloudT& landmarks, size_t step, 
                                  std::vector<int>& lm_idx, Pose3 submap_pose)
{
    // Check we've got the same number of landmarks and indexes
    if(!landmarks.points.size() == lm_idx.size()){
        ROS_ERROR("Different number of landmarks and indexes");
    }

    // Convert landmarks PCL points to gtsam Point3
    int i = 0;
    bool lc_detected = false;
    for (PointT &point : landmarks){
        // If known association is on, the landmarks are already in map frame
        Point3 lm = Vector3f(point.x, point.y, point.z).cast<double>();
        
        graph_->emplace_shared<BearingRangeFactor<Pose3, Point3> >(
            Symbol('x', step+1), Symbol('l', lm_idx.at(i)), 
            submap_pose.bearing(lm), submap_pose.range(lm), brNoise_);

        // Add initial estimate for landmarks
        if (!initValues_->exists(Symbol('l', lm_idx.at(i)))) {
            // Point3 mapLandmark = submap_pose.transformFrom(lm);
            initValues_->insert(Symbol('l', lm_idx.at(i)), lm);
        }
        else{
            lc_detected = true;
            // std::cout << "LC with landmark " << lm_idx.at(i) << std::endl;
            // TODO: if loop closure detected, updateISAM2()
        }
        i++;
    }

    // initValues_->print("Init values ");
    ROS_INFO("RB factor added");
} 

void samGraph::updateISAM2()
{
    isam_->update(*graph_, *initValues_);

    // Values estimate = isam_->calculateEstimate();
    initValues_->print("Init estimate: ");
    graph_.reset(new NonlinearFactorGraph());
    initValues_.reset(new Values());
    ROS_INFO("ISAM updated");
}


//==========================================================================
//==========================================================================
BathySlamNode::BathySlamNode(std::string node_name, ros::NodeHandle &nh) : node_name_(node_name),
                                                                           nh_(&nh)
{

    std::string pings_top, debug_pings_top, odom_top, synch_top, submap_top, indexes_top;
    nh_->param<std::string>("mbes_pings", pings_top, "/gt/mbes_pings");
    nh_->param<std::string>("odom_topic", odom_top, "/gt/odom");
    nh_->param<std::string>("map_frame", map_frame_, "map");
    nh_->param<std::string>("odom_frame", odom_frame_, "odom");
    nh_->param<std::string>("base_link", base_frame_, "base_frame");
    nh_->param<std::string>("mbes_link", mbes_frame_, "mbes_frame");
    nh_->param<std::string>("synch_topic", synch_top, "slam_synch");
    nh_->param<std::string>("submaps_topic", submap_top, "/submaps");
    nh_->param<std::string>("landmarks_idx_topic", indexes_top, "/lm_idx");

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
    submaps_cnt_ = 0;
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
    // current_pose.print("Current pose ");
    current_pose.print("Current pose ");
    odom_step.print("odom_step  ");
    graph_solver->addOdomFactor(current_pose, odom_step, submaps_cnt_);

    // Add landmarks to graph as 3D range-bearing measurements from the current DR pose
    std::vector<int> lm_idx_vec = lm_idx->idx.data;
    PointCloudT lm_pcl;
    pcl::fromROSMsg(*lm_pcl_msg, lm_pcl);
    graph_solver->addLandmarksFactor(lm_pcl, submaps_cnt_, lm_idx_vec, current_pose);

    // If landmarks have been revisited, add measurements to graph and update ISAM
    // if (submaps_cnt_ == 2){
    //     graph_solver->updateISAM2();
    // }

    submaps_cnt_++;
}

Pose3 BathySlamNode::odomStep(double t_step, Pose3 &current_pose)
{
    // Find odom msg corresponding to submap at time t_step
    auto current_odom = odomVec_.begin();
    current_odom = std::find_if(current_odom, odomVec_.end(), [&](const nav_msgs::Odometry& odom) {
        return std::abs(odom.header.stamp.toSec() - t_step) < 0.01;
    });
    if(current_odom == odomVec_.end()){
        std::cout << "Not found current odom that matches submap" << std::endl;
    }

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
    double step = pos_step.norm() * cos(euler_step[2]);

    // Store odom from current submap as prev
    prev_submap_odom_.reset(new nav_msgs::Odometry(*current_odom));

    // TODO: clear odomVec_ once in a while

    // Convert current_odom to gtsam pose for landmarks computation
    Eigen::Vector3d euler_now = q_now.normalized().toRotationMatrix().eulerAngles(0, 1, 2);
    current_pose = Pose3(Rot3::Ypr(euler_now[2], euler_now[1], euler_now[0]), 
                         Point3(pos_now[0], pos_now[1], pos_now[2]));

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
