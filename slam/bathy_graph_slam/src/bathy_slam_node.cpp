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
        tflistener_.waitForTransform(base_frame_, mbes_frame_, ros::Time(0), ros::Duration(10.0) );
        tflistener_.lookupTransform(base_frame_, mbes_frame_, ros::Time(0), tf_base_mbes_);
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
    graph_solver.reset(new samGraph());

    // Empty service to synch the applications waiting for this node to start
    synch_service_ = nh_->advertiseService(synch_top,
                                            &BathySlamNode::emptySrv, this);
    ROS_INFO("Initialized SLAM");
}

bool BathySlamNode::emptySrv(std_srvs::EmptyRequest &req, std_srvs::EmptyResponse &res)
{
    return true;
}

void BathySlamNode::updateTf()
{
    geometry_msgs::TransformStamped msg_map_submap;
    tf::StampedTransform tf_map_submap_stp(tf_submaps_vec_.at(submaps_cnt_),
                                            ros::Time::now(),
                                            odom_frame_,
                                            "submap_" + std::to_string(submaps_cnt_));

    std::cout << "submap_" + std::to_string(submaps_cnt_) << std::endl;
    tf::transformStampedTFToMsg(tf_map_submap_stp, msg_map_submap);
    static_broadcaster_.sendTransform(msg_map_submap);

    // For RVIZ
    // sensor_msgs::PointCloud2 submap_msg;
    // pcl::toROSMsg(submaps_vec_.at(submaps_cnt_).submap_pcl_, submap_msg);
    // submap_msg.header.frame_id = "submap_" + std::to_string(submaps_cnt_);
    // submap_msg.header.stamp = ros::Time::now();
    // submaps_pub_.publish(submap_msg);
}

void BathySlamNode::pingCB(const sensor_msgs::PointCloud2Ptr &mbes_ping, 
                           const nav_msgs::OdometryPtr &odom_msg)
{
    // Set prior on first odom pose
    if (first_msg_){
        graph_solver->addPrior();
        first_msg_ = false;  
    }

    tf::Transform odom_base_tf;
    tf::poseMsgToTF(odom_msg->pose.pose, odom_base_tf);
    // Storing point cloud of mbes pings in mbes frame and current tf_odom_mbes_
    submap_raw_.emplace_back(mbes_ping, odom_base_tf * tf_base_mbes_);

    if (submap_raw_.size() > 100)
    {
        this->addSubmap(submap_raw_);
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
    ROS_INFO("Checking for loop closures ");
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

std::tuple<double, double> BathySlamNode::extractLandmarks(SubmapObj& submap_i){
    // Parameters for sift computation
    const float min_scale = 0.01f;
    const int n_octaves = 6;
    const int n_scales_per_octave = 4;
    const float min_contrast = 0.005f;

    // // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>(submap_i.submap_pcl_));
    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    // pcl::io::loadPCDFile<pcl::PointXYZ>("/home/torroba/test_pcd.pcd", *cloud_xyz);
    // std::cout << "Copied pcl " << cloud_xyz->size() << std::endl;

    // // Estimate the sift interest points using z values from xyz as the Intensity variants
    // pcl::SIFTKeypoint<pcl::PointXYZ, pcl::PointWithScale> sift;
    // pcl::PointCloud<pcl::PointWithScale> result;
    // pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    // sift.setSearchMethod(tree);
    // sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    // sift.setMinimumContrast(min_contrast);
    // sift.setInputCloud(cloud_xyz);
    // std::cout << "SIFT set up " << std::endl;
    // sift.compute(result);

    // std::cout << "No of SIFT points in the result are " << result.size() << std::endl;

    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
    // copyPointCloud(result, *cloud_temp);

    // sensor_msgs::PointCloud2 submap_msg;
    // pcl::toROSMsg(*cloud_temp, submap_msg);
    // submap_msg.header.frame_id = "submap_" + std::to_string(submaps_cnt_);
    // submap_msg.header.stamp = ros::Time::now();
    // submaps_pub_.publish(submap_msg);

    return std::make_tuple(1.0, 1.0);
}

void BathySlamNode::addSubmap(std::vector<ping_raw> submap_pings)
{
    std::cout << "Calling submap constructor" << std::endl;

    // Store submap tf
    tf::Transform tf_submap_i = std::get<1>(submap_pings.at(submap_pings.size()/2));
    tf_submaps_vec_.push_back(tf_submap_i);

    // Create submap object
    SubmapObj submap_i;
    Eigen::Affine3d tf_affine;
    tf::transformTFToEigen(tf_submap_i, tf_affine);
    submap_i.submap_tf_ = tf_affine.matrix().cast<float>();

    for (ping_raw &ping_j : submap_pings)
    {
        PointCloudT pcl_ping;
        pcl::fromROSMsg(*std::get<0>(ping_j).get(), pcl_ping);
        pcl_ros::transformPointCloud(pcl_ping, pcl_ping, tf_submap_i.inverse() * std::get<1>(ping_j));
        submap_i.submap_pcl_ += pcl_ping;
    }

    submap_i.submap_pcl_.sensor_origin_ << submap_i.submap_tf_.translation();
    submap_i.submap_pcl_.sensor_orientation_ = submap_i.submap_tf_.linear();
    submap_i.submap_id_ = submaps_cnt_;

    submap_i.submap_pcl_.width = submap_i.submap_pcl_.size();
    submap_i.submap_pcl_.height = 1;
    submap_i.submap_pcl_.is_dense = true;

    // For testing outside the environment
    // for (const auto &point : submap_i.submap_pcl_)
    //     std::cerr << "    " << point.x << " " << point.y << " " << point.z << std::endl;

    pcl::io::savePCDFileASCII("/home/torroba/test_pcd.pcd", submap_i.submap_pcl_);

    // submap_i.auv_tracks_ = submap_i.submap_tf_.translation().transpose().cast<double>();
    submap_i.auv_tracks_ =Eigen::MatrixXd(3, 3);

    // Extract landmarks from submap point cloud
    std::tuple<double, double> meas_vec = this->extractLandmarks(submap_i);

    // Check for loop closures
    // if(submaps_cnt_ > 1){
    //     this->checkForLoopClosures(submap_i);
    // }
    // std::cout << "Checked for LCs" << std::endl;

    // Store submap
    submaps_vec_.push_back(submap_i);

    // Update graph DR
    if(submaps_cnt_ > 0){
        Pose2 odom_step = this->odomStep(submaps_cnt_);
        graph_solver->addOdomFactor(odom_step, submaps_cnt_);
    }
    std::cout << "Odom added" << std::endl;

    // // Update TF
    this->updateTf();

    submaps_cnt_++;
}

namespace pcl
{
    template <>
    struct SIFTKeypointFieldSelector<PointXYZ>
    {
        inline float
        operator()(const PointXYZ &p) const
        {
            return p.z;
        }
    };
}

int main(int argc, char** argv){

    ros::init(argc, argv, "bathy_slam_node");
    ros::NodeHandle nh("~");

    // Parameters for sift computation
    const float min_scale = 1.0f;
    const int n_octaves = 6;
    const int n_scales_per_octave = 4;
    const float min_contrast = 0.005f;

    // pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>(submap_i.submap_pcl_));
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ>("/home/torroba/test_pcd.pcd", *cloud_xyz);
    std::cout << "Copied pcl " << cloud_xyz->size() << std::endl;

    // Estimate the sift interest points using z values from xyz as the Intensity variants
    pcl::SIFTKeypoint<pcl::PointXYZ, pcl::PointWithScale> sift;
    pcl::PointCloud<pcl::PointWithScale> result;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    sift.setSearchMethod(tree);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(cloud_xyz);
    std::cout << "SIFT set up " << std::endl;
    sift.compute(result);

    std::cout << "No of SIFT points in the result are " << result.size() << std::endl;

    // Copying the pointwithscale to pointxyz so as visualize the cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
    copyPointCloud(result, *cloud_temp);
    std::cout << "SIFT points in the result are " << cloud_temp->size() << std::endl;
    // Visualization of keypoints along with the original cloud
    pcl::visualization::PCLVisualizer viewer("PCL Viewer");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler(cloud_temp, 0, 255, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color_handler(cloud_xyz, 255, 0, 0);
    viewer.setBackgroundColor(0.0, 0.0, 0.0);
    viewer.addPointCloud(cloud_xyz, cloud_color_handler, "cloud");
    viewer.addPointCloud(cloud_temp, keypoints_color_handler, "keypoints");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 7, "keypoints");

    while (!viewer.wasStopped())
    {
        viewer.spinOnce();
    }

    return 0;

   
    // BathySlamNode* bathy_slam = new BathySlamNode(ros::this_node::getName(), nh);

    // // ros::Timer timer = nh.createTimer(ros::Duration(1), &BathySlamNode::bcMapSubmapsTF, bathy_slam);

    // ros::spin();
    // ros::waitForShutdown();

    // if(!ros::ok()){
    //     delete bathy_slam;
    // }
    // ROS_INFO("Bathy SLAM node finished");

    return 0;
}
