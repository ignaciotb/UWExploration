#include <bathy_graph_slam/bathy_slam.hpp>
// #include <pcl/keypoints/sift_keypoint.h>

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
    // Add a prior at time 0 and update isam
    graph_->addPrior(0, Pose2(0.0, 0.0, 0.0), odoNoise_);
    initValues_->insert((0), Pose2(0.01, 0.01, 0.01));

    isam_->update(*graph_, *initValues_);
    graph_.reset(new NonlinearFactorGraph());
    initValues_.reset(new Values());
    ROS_INFO("Prior updated");
}

void samGraph::addOdomFactor(Pose2 odom_step, size_t step)
{
    // Add odometry
    graph_->push_back(BetweenFactor<Pose2>(step - 1, step, odom_step, odoNoise_));

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
BathySlamNode::BathySlamNode(std::string node_name, ros::NodeHandle &nh): node_name_(node_name), nh_(&nh)
{

    std::string pings_top, debug_pings_top, odom_top, enable_top, synch_top, submap_top;
    nh_->param<std::string>("mbes_pings", pings_top, "/gt/mbes_pings");
    nh_->param<std::string>("odom_gt", odom_top, "/gt/odom");
    nh_->param<std::string>("map_frame", map_frame_, "map");
    nh_->param<std::string>("odom_frame", odom_frame_, "odom");
    nh_->param<std::string>("base_link", base_frame_, "base_frame");
    nh_->param<std::string>("mbes_link", mbes_frame_, "mbes_frame");
    nh_->param<std::string>("survey_finished_top", enable_top, "enable");
    nh_->param<std::string>("synch_topic", synch_top, "slam_synch");
    nh_->param<std::string>("submaps_topic", submap_top, "/submaps");

    enable_subs_ = nh_->subscribe(enable_top, 1, &BathySlamNode::enableCB, this);
    submap_subs_ = nh_->subscribe(submap_top, 2, &BathySlamNode::submapCB, this);

    try {
        tflistener_.waitForTransform(base_frame_, mbes_frame_, ros::Time(0), ros::Duration(10.0) );
        tflistener_.lookupTransform(base_frame_, mbes_frame_, ros::Time(0), tf_base_mbes_);
        ROS_INFO_NAMED(node_name_, " locked transform base --> sensor");
    }
    catch(tf::TransformException &exception) {
        ROS_ERROR("%s", exception.what());
        ros::Duration(1.0).sleep();
    }

    // Initialize survey params
    first_msg_ = true;
    submaps_cnt_ = 0;

    // ISAM solver
    graph_solver.reset(new samGraph());

    // Empty service to synch the applications waiting for this node to start
    synch_service_ = nh_->advertiseService(synch_top,
                                            &BathySlamNode::emptySrv, this);
    ROS_INFO_NAMED(node_name_, " initialized");
}

bool BathySlamNode::emptySrv(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res)
{
    return true;
}

void BathySlamNode::submapCB(const sensor_msgs::PointCloud2Ptr &submap_i)
{
    // Set prior on first odom pose
    if (first_msg_){
        graph_solver->addPrior();
        first_msg_ = false;  
    }

    // Check for loop closures
    // if(submaps_cnt_ > 1){
    //     this->checkForLoopClosures(submap_i);
    // }
    // std::cout << "Checked for LCs" << std::endl;

    // Update graph DR
    if (submaps_cnt_ > 0)
    {
        Pose2 odom_step = this->odomStep(submaps_cnt_);
        graph_solver->addOdomFactor(odom_step, submaps_cnt_);
    }
    std::cout << "Odom added" << std::endl;

    submaps_cnt_++;
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

Pose2 BathySlamNode::odomStep(int odom_step)
{
    // Rotation
    Eigen::Quaternionf q_prev(submaps_vec_.at(submaps_cnt_ - 1).submap_tf_.linear());
    Eigen::Quaternionf q_now(submaps_vec_.at(submaps_cnt_).submap_tf_.linear()); 
    Eigen::Quaternionf q_step = q_now.normalized() * q_prev.inverse().normalized();
    Eigen::Vector3f euler = q_step.toRotationMatrix().eulerAngles(0, 1, 2);

    // // Translation
    Eigen::Vector3f pos_now, pos_prev;
    pos_prev = submaps_vec_.at(submaps_cnt_ - 1).submap_tf_.translation();
    pos_now = submaps_vec_.at(submaps_cnt_).submap_tf_.translation();
    Eigen::Vector3f pos_step = pos_now - pos_prev;
    std::cout << pos_step << euler << std::endl;

    return Pose2(pos_step[0], pos_step[1], euler[2]);
}

void BathySlamNode::checkForLoopClosures(SubmapObj submap_i){
    // Look for submaps overlapping the latest one
    ROS_INFO_NAMED(node_name_, " Checking for loop closures ");
    SubmapsVec submaps_prev;
    for (SubmapObj &submap_k : submaps_vec_)
    {
        // Don't look for overlaps between submaps of the same swath or the prev submap
        if (submap_k.submap_id_ != submap_i.submap_id_ - 1)
        {
            submaps_prev.push_back(submap_k);
        }
    }
    std::cout << "Stored prev potential submaps " << submaps_prev.size() << std::endl;

    submap_i.findOverlaps(submaps_prev);
    ROS_INFO("Done looking for overlaps");
    submaps_prev.clear();
    for (unsigned int j = 0; j < submap_i.overlaps_idx_.size(); j++)
    {
        std::cout << "Overlap with submap " << submap_i.overlaps_idx_.at(j);
    }
    ROS_INFO("Done checking for LCs");
}

int main(int argc, char** argv){

    ros::init(argc, argv, "bathy_slam_node");
    ros::NodeHandle nh("~");

    boost::shared_ptr<BathySlamNode> bathySlam;
    bathySlam.reset(new BathySlamNode(ros::this_node::getName(), nh));

    // ros::Timer timer = nh.createTimer(ros::Duration(1), &BathySlamNode::bcMapSubmapsTF, bathySlam);

    ros::spin();

    ROS_INFO_NAMED(ros::this_node::getName(), " finished");

    return 0;
}
