#include "auv_2_ros/mbes_meas.hpp"

MbesMeas::MbesMeas(std::string node_name, ros::NodeHandle &nh):
    node_name_(node_name), nh_(&nh){

    std::string gt_pings_top, debug_pings_top, map_top, gt_odom_top;
    nh_->param<std::string>("world_frame", world_frame_, "world");
    nh_->param<std::string>("map_frame", map_frame_, "map");
    nh_->param<std::string>("mbes_link", mbes_frame_, "mbes_link");
    nh_->param<std::string>("map_pcl", map_top, "/map");
    nh_->param<double>("mbes_open_angle", mbes_opening_, 1.5708);
    nh_->param<double>("num_beams_sim", n_beams_, 100);

    map_pub_ = nh_->advertise<sensor_msgs::PointCloud2>(map_top, 10);

    // Create and start Sim MBES as
    as_ = new actionlib::SimpleActionServer<auv_2_ros::MbesSimAction>(*nh_, node_name_,
                                                                      boost::bind(&MbesMeas::measCB,
                                                                                  this, _1), false);
    as_->start();
}


MbesMeas::~MbesMeas(){

}

void MbesMeas::init(const boost::filesystem::path map_path){

     // Read map
    MapObj map_loc;
    std_data::pt_submaps ss = std_data::read_data<std_data::pt_submaps>(map_path);
    std::tie(map_loc, map_tf_)= parseMapAUVlib(ss);
    maps_gt_.push_back(map_loc);

    // Create voxel grid with the bathymetric map
    vox_oc_.setLeafSize(10,10,10);
    vox_oc_.initializeVoxelGrid(maps_gt_.at(0));

    // Publish voxelized map once
    sensor_msgs::PointCloud2 map;
    PointCloudT pcl_filtered = vox_oc_.getFilteredPointCloud();
    pcl::toROSMsg(pcl_filtered, map);
    map.header.frame_id = map_frame_;
    map_pub_.publish(map);

    ROS_INFO("Initialized MBES simulation");
}


void MbesMeas::measCB(const auv_2_ros::MbesSimGoalConstPtr &mbes_goal){

    // Create simulated ping
    Eigen::Isometry3d sensor_tf;
    tf::transformMsgToEigen(mbes_goal->mbes_pose.transform, sensor_tf);
    Eigen::Isometry3f tf = sensor_tf.inverse().cast<float>();
    vox_oc_.createMBES(mbes_opening_, n_beams_, tf);

    PointCloudT::Ptr sim_mbes_i_pcl(new PointCloudT);
    vox_oc_.pingComputation(*sim_mbes_i_pcl);
    pcl_ros::transformPointCloud(*sim_mbes_i_pcl, *sim_mbes_i_pcl, mbes_goal->mbes_pose.transform);
    if(sim_mbes_i_pcl->points.size() == 0){
        ROS_WARN("No multibeam hits! You're out of the GT map");
    }
    sensor_msgs::PointCloud2 sim_ping;
    pcl::toROSMsg(*sim_mbes_i_pcl.get(), sim_ping);
    sim_ping.header.frame_id = mbes_goal->mbes_pose.header.frame_id;
    sim_ping.header.stamp = mbes_goal->mbes_pose.header.stamp;

    result_.sim_mbes = sim_ping;
    as_->setSucceeded(result_);
}


void MbesMeas::broadcastW2MTf(const ros::TimerEvent&){

    // Publish world-->map frames
    geometry_msgs::TransformStamped w2m_static_tfStamped;

    w2m_static_tfStamped.header.stamp = ros::Time::now();
    w2m_static_tfStamped.header.frame_id = world_frame_;
    w2m_static_tfStamped.child_frame_id = map_frame_;
    w2m_static_tfStamped.transform.translation.x = map_tf_.translation()[0];
    w2m_static_tfStamped.transform.translation.y = map_tf_.translation()[1];
    w2m_static_tfStamped.transform.translation.z = map_tf_.translation()[2];
    tf2::Quaternion quatw2m;
    Eigen::Vector3d euler = map_tf_.linear().matrix().eulerAngles(0, 1, 2);
    quatw2m.setRPY(euler[0], euler[1], euler[2]);
    w2m_static_tfStamped.transform.rotation.x = quatw2m.x();
    w2m_static_tfStamped.transform.rotation.y = quatw2m.y();
    w2m_static_tfStamped.transform.rotation.z = quatw2m.z();
    w2m_static_tfStamped.transform.rotation.w = quatw2m.w();
    static_broadcaster_.sendTransform(w2m_static_tfStamped);
}
