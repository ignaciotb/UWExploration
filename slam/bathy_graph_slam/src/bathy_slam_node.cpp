#include <bathy_graph_slam/bathy_slam.hpp>

samGraph::samGraph()
{
    // ISAM solver params
    ISAM2Params params = ISAM2Params(ISAM2GaussNewtonParams(0.001), 0.0,
                                     0, false, true,
                                     ISAM2Params::CHOLESKY, true,
                                     DefaultKeyFormatter, true);

    // TODO: this has to be an input param
    odoNoise_ = noiseModel::Diagonal::Sigmas((Vector(3) << 0.1, 0.1, M_PI / 100.0).finished());
    brNoise_ = noiseModel::Diagonal::Sigmas((Vector(2) << M_PI / 100.0, 0.1).finished());

    // Instantiate pointers
    isam_.reset(new ISAM2(params));
    graph_.reset(new NonlinearFactorGraph());
    initValues_.reset(new Values());
}

samGraph::~samGraph()
{
}

void samGraph::addPrior()
{
    // TODO: make Pose3
    // Add a prior at time 0 and update isam
    graph_->addPrior(0, Pose2(0.0, 0.0, 0.0), odoNoise_);
    initValues_->insert((0), Pose2(0.0, 0.0, 0.0));

    // TODO: do we need to update isam here?
    isam_->update(*graph_, *initValues_);
    graph_.reset(new NonlinearFactorGraph());
    initValues_.reset(new Values());
    ROS_INFO("Prior updated");
}

void samGraph::addOdomFactor(Pose2 odom_step, size_t step)
{
    // Add odometry
    // submap i will be DR factor i+1 since the origin 
    // (where there's no submap) is the factor 0
    graph_->push_back(BetweenFactor<Pose2>(step, step+1, odom_step, odoNoise_));

    // Predict pose and add as initial estimate
    Pose2 predictedPose = lastPose_.compose(odom_step);
    lastPose_ = predictedPose;
    initValues_->insert((step), predictedPose);
    ROS_INFO("Odom factor added");
}

void samGraph::addRangeFactor(Pose2 odom_step, size_t step, int lm_idx)
{
    graph_->push_back(BearingRangeFactor<Pose2, Point2>(step, lm_idx,
                                                        Rot2::fromAngle(M_PI / 4.0), 5.0, brNoise_));

    initValues_->insert(lm_idx, Point2(5.0 / sqrt(2.0), 5.0 / sqrt(2.0)));
    ROS_INFO("RB factor added");
}

void samGraph::updateISAM2()
{
    isam_->update(*graph_, *initValues_);
    graph_.reset(new NonlinearFactorGraph());
    initValues_.reset(new Values());
    ROS_INFO("ISAM updated");
}


//==========================================================================
//==========================================================================
BathySlamNode::BathySlamNode(std::string node_name, ros::NodeHandle &nh) : node_name_(node_name),
                                                                           nh_(&nh), tfListener_(tfBuffer_)
{

    std::string pings_top, debug_pings_top, odom_top, synch_top, submap_top;
    nh_->param<std::string>("mbes_pings", pings_top, "/gt/mbes_pings");
    nh_->param<std::string>("odom_topic", odom_top, "/gt/odom");
    nh_->param<std::string>("map_frame", map_frame_, "map");
    nh_->param<std::string>("odom_frame", odom_frame_, "odom");
    nh_->param<std::string>("base_link", base_frame_, "base_frame");
    nh_->param<std::string>("mbes_link", mbes_frame_, "mbes_frame");
    nh_->param<std::string>("synch_topic", synch_top, "slam_synch");
    nh_->param<std::string>("submaps_topic", submap_top, "/submaps");

    submap_subs_ = nh_->subscribe(submap_top, 2, &BathySlamNode::updateGraphCB, this);
    odom_subs_ = nh_->subscribe(odom_top, 20, &BathySlamNode::odomCB, this);

    // tfListener_.reset(new tf2_ros::TransformListener(tfBuffer_));
    // try {

    //     geometry_msgs::TransformStamped tf_base_mbes;
    //     // tfBuffer_.waitForTransform(base_frame_, mbes_frame_, ros::Time(0), ros::Duration(10.0));
    //     tf_base_mbes = tfBuffer_.lookupTransform(base_frame_, mbes_frame_, ros::Time(0));
    //     tf::transformStampedMsgToTF(tf_base_mbes, tf_base_mbes_);
    //     ROS_INFO_NAMED(node_name_, " locked transform base --> sensor");
    // }
    // catch(tf::TransformException &exception) {
    //     ROS_ERROR("%s", exception.what());
    //     ros::Duration(1.0).sleep();
    // }

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

void BathySlamNode::updateGraphCB(const sensor_msgs::PointCloud2Ptr &pcl_msg)
{
    // ROS_INFO_NAMED("SLAM ", "submap received");
    // geometry_msgs::TransformStamped tf_o_submapi;
    // tf_o_submapi = tfBuffer_.lookupTransform(odom_frame_, pcl_msg->header.frame_id, ros::Time(0));
    // Eigen::Isometry3d submap_tf_d;
    // tf::transformMsgToEigen(tf_o_submapi.transform, submap_tf_d);
    // ROS_INFO_NAMED("SLAM ", "TF received");

    // Set prior on first odom pose
    if (first_msg_)
    {
        graph_solver->addPrior();
        first_msg_ = false;
    }

    // Update graph DR concatenating odom msgs between submaps
    Pose2 odom_step = this->odomStep(pcl_msg->header.stamp.toSec());
    graph_solver->addOdomFactor(odom_step, submaps_cnt_);

    // If landmarks have been revisited, add measurements to graph and update ISAM
    // if (loop_closure){
    //     graph_solver->updateISAM2();
    // }

    submaps_cnt_++;
}

Pose2 BathySlamNode::odomStep(double t_step)
{
    // Find odom msg corresponding to submap at time t_step
    auto pos = odomVec_.begin();
    pos = std::find_if(pos, odomVec_.end(), [&](const nav_msgs::Odometry& odom) {
        return odom.header.stamp.toSec() == t_step;
    });

    // Odom step as diff between odom_msgs from two consecutive submaps
    Eigen::Affine3d pose_now, pose_prev;
    tf::poseMsgToEigen (pos->pose.pose, pose_now);
    tf::poseMsgToEigen (prev_submap_odom_->pose.pose, pose_prev);

    // Rotation
    Eigen::Quaterniond q_prev(pose_prev.linear());
    Eigen::Quaterniond q_now(pose_now.linear()); 
    Eigen::Quaterniond q_step = q_now.normalized() * q_prev.inverse().normalized();
    Eigen::Vector3d euler = q_step.toRotationMatrix().eulerAngles(0, 1, 2);

    // Translation
    Eigen::Vector3d pos_now, pos_prev;
    pos_prev = pose_prev.translation();
    pos_now = pose_now.translation();
    Eigen::Vector3d pos_step = pos_now - pos_prev;

    // Store odom from current submap as prev
    prev_submap_odom_.reset(new nav_msgs::Odometry(*pos));

    // TODO: clear odomVec_ once in a while

    return Pose2(pos_step[0], pos_step[1], euler[2]);
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
