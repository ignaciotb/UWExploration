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

void samGraph::addPrior()
{
    // Add a prior at time 0 and update isam
    graph_->addPrior(Symbol('x', 0), Pose3(Rot3(), Point3(0.0, 0.0, 0.0)), odoNoise_);
    initValues_->insert(Symbol('x', 0), Pose3(Rot3(), Point3(0.0, 0.0, 0.0)));
    // Init last pose where the odom frame is
    lastPose_ = Pose3(Rot3(), Point3(0.0, 0.0, 0.0));

    // TODO: do we need to update isam here?
    // isam_->update(*graph_, *initValues_);
    // graph_.reset(new NonlinearFactorGraph());
    // initValues_.reset(new Values());
    ROS_INFO("Prior updated");
}

void samGraph::addOdomFactor(Pose3 odom_step, size_t step)
{
    // Add odometry
    // submap i will be DR factor i+1 since the origin 
    // (where there's no submap) is the factor 0
    graph_->emplace_shared<BetweenFactor<Pose3> >(Symbol('x', step), Symbol('x', step+1),
                                                  odom_step, odoNoise_);

    // Predict pose and add as initial estimate
    Pose3 predictedPose = lastPose_.compose(odom_step);
    lastPose_ = predictedPose;
    initValues_->insert(Symbol('x', step+1), predictedPose);
    ROS_INFO("Odom factor added");
}

void samGraph::addLandmarksFactor(PointCloudT& landmarks, size_t step, 
                                  std::vector<int>& lm_idx, Pose3 submap_pose)
{
    // TODO: handle landmarks id
    // Convert landmarks PCL points to gtsam Point3
    for (PointT &point : landmarks){
        Point3 lm = Vector3f(point.x, point.y, point.z).cast<double>();
        graph_->emplace_shared<BearingRangeFactor<Pose3, Point3> >(
            Symbol('x', step+1), Symbol('l', num_landmarks_), submap_pose.bearing(lm), submap_pose.range(lm), brNoise_);

        // Add initial estimate for landmarks
        if (!initValues_->exists(Symbol('l', num_landmarks_))) {
            // TODO: transform landmarks to map frame. Currently in odom frame
            // Pose3 submap_in_map(submap_pose.matrix() * submap_pose.matrix());
            Point3 mapLandmark = submap_pose.transformFrom(lm);
            initValues_->insert(Symbol('l', num_landmarks_), mapLandmark);
        }
        num_landmarks_++;
    }
    // initValues_->print("Init values ");
    ROS_INFO("RB factor added");
}

void samGraph::updateISAM2()
{
    ROS_INFO("About to update ISAM");
    isam_->update(*graph_, *initValues_);

    Values estimate = isam_->calculateEstimate();
    estimate.print("Current estimate: ");
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

void BathySlamNode::updateGraphCB(const sensor_msgs::PointCloud2Ptr &submap_meas)
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
    Pose3 current_pose;
    Pose3 odom_step = this->odomStep(submap_meas->header.stamp.toSec(), current_pose);
    graph_solver->addOdomFactor(odom_step, submaps_cnt_);

    std::vector<int> lm_idx;
    PointCloudT meas_pcl;
    pcl::fromROSMsg(*submap_meas, meas_pcl);
    graph_solver->addLandmarksFactor(meas_pcl, submaps_cnt_, lm_idx, current_pose);

    // If landmarks have been revisited, add measurements to graph and update ISAM
    if (submaps_cnt_ == 2){
        graph_solver->updateISAM2();
    }

    submaps_cnt_++;
}

Pose3 BathySlamNode::odomStep(double t_step, Pose3 &current_pose)
{
    // Find odom msg corresponding to submap at time t_step
    auto current_odom = odomVec_.begin();
    current_odom = std::find_if(current_odom, odomVec_.end(), [&](const nav_msgs::Odometry& odom) {
        return odom.header.stamp.toSec() == t_step;
    });

    // Odom step as diff between odom_msgs from two consecutive submaps
    Eigen::Affine3d pose_now, pose_prev;
    tf::poseMsgToEigen (current_odom->pose.pose, pose_now);
    tf::poseMsgToEigen (prev_submap_odom_->pose.pose, pose_prev);

    // Rotation
    Eigen::Quaterniond q_prev(pose_prev.linear());
    Eigen::Quaterniond q_now(pose_now.linear()); 
    Eigen::Quaterniond q_step = q_now.normalized() * q_prev.inverse().normalized();
    Eigen::Vector3d euler_step = q_step.toRotationMatrix().eulerAngles(0, 1, 2);

    // Translation
    Eigen::Vector3d pos_now, pos_prev;
    pos_prev = pose_prev.translation();
    pos_now = pose_now.translation();
    Eigen::Vector3d pos_step = pos_now - pos_prev;

    // Store odom from current submap as prev
    prev_submap_odom_.reset(new nav_msgs::Odometry(*current_odom));

    // TODO: clear odomVec_ once in a while

    // Convert current_odom to gtsam pose for landmarks computation
    Eigen::Vector3d euler_now = q_now.normalized().toRotationMatrix().eulerAngles(0, 1, 2);
    current_pose = Pose3(Rot3::Ypr(euler_now[2], euler_now[1], euler_now[0]), 
                         Point3(pos_now[0], pos_now[1], pos_now[2]));

    // Return odom step as gtsam pose
    return Pose3(Rot3::Ypr(euler_step[2], euler_step[1], euler_step[0]), 
                Point3(pos_step[0], pos_step[1], pos_step[2]));
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
