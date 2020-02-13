#include "bathy_construction/bathymap_constructor.hpp"


BathymapConstructor::BathymapConstructor(std::string node_name, ros::NodeHandle &nh):
    node_name_(node_name), nh_(&nh){

    ping_pub_ = nh_->advertise<sensor_msgs::PointCloud2>("mbes_pings", 10);
    test_pub_ = nh_->advertise<sensor_msgs::PointCloud2>("debug", 10);
    map_pub_ = nh_->advertise<sensor_msgs::PointCloud2>("map", 10);

    tfListener_ = new tf2_ros::TransformListener(tfBuffer_);

    ping_num_ = 0;

    ROS_INFO("Created bathymap constructor");
}

BathymapConstructor::~BathymapConstructor(){

}


void BathymapConstructor::init(const boost::filesystem::path map_path, const boost::filesystem::path auv_path){

    // Read map
    MapObj map_loc;
    Eigen::Isometry3d map_tf;
    std_data::pt_submaps ss = std_data::read_data<std_data::pt_submaps>(map_path);
    std::tie(map_loc, map_tf)= parseMapAUVlib(ss);
    maps_gt_.push_back(map_loc);

    // Read pings
    std_data::mbes_ping::PingsT std_pings = std_data::read_data<std_data::mbes_ping::PingsT>(auv_path);
    std::cout << "Number of pings " << std_pings.size() << std::endl;
    traj_pings_ = parsePingsAUVlib(std_pings, map_tf);

    // Filtering of maps
    PointCloudT::Ptr cloud_ptr (new PointCloudT);
    pcl::UniformSampling<PointT> us_filter;
    us_filter.setInputCloud (cloud_ptr);
    us_filter.setRadiusSearch(2);   // Tune for speed/map resolution
    for(SubmapObj& submap_i: maps_gt_){
        *cloud_ptr = submap_i.submap_pcl_;
        us_filter.setInputCloud(cloud_ptr);
        us_filter.filter(*cloud_ptr);
        submap_i.submap_pcl_ = *cloud_ptr;
    }

    map_tf_ = map_loc.submap_tf_;
    odom_tf_ = traj_pings_.at(0).submap_tf_;

    ROS_INFO("Initialized bathymap constructor");
}

void BathymapConstructor::broadcastTf(const ros::TimerEvent& event){

    // Publish world-->map-->odom frames
    static tf2_ros::StaticTransformBroadcaster static_broadcaster;
    geometry_msgs::TransformStamped w2m_static_tfStamped, m2o_static_tfStamped;
    static tf2_ros::TransformBroadcaster br;

    w2m_static_tfStamped.header.stamp = ros::Time::now();
    w2m_static_tfStamped.header.frame_id = "world";
    w2m_static_tfStamped.child_frame_id = "map";
    w2m_static_tfStamped.transform.translation.x = map_tf_.translation()[0];
    w2m_static_tfStamped.transform.translation.y = map_tf_.translation()[1];
    w2m_static_tfStamped.transform.translation.z = map_tf_.translation()[2];
    tf2::Quaternion quatw2m;
    Eigen::Vector3d euler = map_tf_.linear().matrix().eulerAngles(0, 1, 2).cast<double>();
    quatw2m.setRPY(euler[0], euler[1], euler[2]);
    w2m_static_tfStamped.transform.rotation.x = quatw2m.x();
    w2m_static_tfStamped.transform.rotation.y = quatw2m.y();
    w2m_static_tfStamped.transform.rotation.z = quatw2m.z();
    w2m_static_tfStamped.transform.rotation.w = quatw2m.w();
    static_broadcaster.sendTransform(w2m_static_tfStamped);

    Eigen::Vector3f map_odom = odom_tf_.translation();

    m2o_static_tfStamped.header.stamp = ros::Time::now();
    m2o_static_tfStamped.header.frame_id = "map";
    m2o_static_tfStamped.child_frame_id = "odom";
    m2o_static_tfStamped.transform.translation.x = map_odom[0];
    m2o_static_tfStamped.transform.translation.y = map_odom[1];
    m2o_static_tfStamped.transform.translation.z = map_odom[2];
    euler = odom_tf_.linear().matrix().eulerAngles(0, 1, 2).cast<double>();
    tf2::Quaternion quatm2o;
    quatm2o.setRPY(euler[0], euler[1], euler[2]);
    m2o_static_tfStamped.transform.rotation.x = quatm2o.x();
    m2o_static_tfStamped.transform.rotation.y = quatm2o.y();
    m2o_static_tfStamped.transform.rotation.z = quatm2o.z();
    m2o_static_tfStamped.transform.rotation.w = quatm2o.w();
    static_broadcaster.sendTransform(m2o_static_tfStamped);

    geometry_msgs::TransformStamped new_pings_tf;
    ROS_INFO("Running %d", ping_num_);
    new_pings_tf.header.frame_id = "map";
    new_pings_tf.child_frame_id = "ping_" + std::to_string(ping_num_);
//    new_pings_tf.header.stamp = ros::Time::now();
    Eigen::Vector3f odom_ping_i = traj_pings_.at(ping_num_).submap_tf_.translation()/* - map_odom*/;
//        odom_ping_i = traj_pings_.at(0).submap_tf_.linear().matrix().transpose() * odom_ping_i;
    new_pings_tf.transform.translation.x = odom_ping_i[0];
    new_pings_tf.transform.translation.y = odom_ping_i[1];
    new_pings_tf.transform.translation.z = odom_ping_i[2];
    tf2::Quaternion quato2p;
    euler = (traj_pings_.at(ping_num_).submap_tf_.linear().matrix()).eulerAngles(0, 1, 2).cast<double>();
    quato2p.setRPY(euler[0], euler[1], euler[2]);
    new_pings_tf.transform.rotation.x = quato2p.x();
    new_pings_tf.transform.rotation.y = quato2p.y();
    new_pings_tf.transform.rotation.z = quato2p.z();
    new_pings_tf.transform.rotation.w = quato2p.w();
    pings_tf_.push_back(new_pings_tf);

    for(geometry_msgs::TransformStamped& tf_ping_i: pings_tf_){
        tf_ping_i.header.stamp = ros::Time::now();
        br.sendTransform(tf_ping_i);
    }
    ROS_INFO("Broadcasting tf");

    this->run();

}

void BathymapConstructor::run(){

    // Publish MBES pings
    sensor_msgs::PointCloud2 mbes_i, mbes_i_map, map;
    PointCloudT::Ptr mbes_i_pcl(new PointCloudT);
    PointCloudT::Ptr mbes_i_pcl_map(new PointCloudT);

    // Publish map
    if(ping_num_ ==0 ){
        pcl::toROSMsg(maps_gt_.at(0).submap_pcl_, map);
        map.header.frame_id = "map";
        map_pub_.publish(map);
    }

    // Original ping (for debugging)

    // Transform points from map to ping i
    geometry_msgs::TransformStamped transformStamped;
    try{
        tfBuffer_.canTransform("ping_"+std::to_string(ping_num_), "map", ros::Time(0), ros::Duration(1));
        transformStamped = tfBuffer_.lookupTransform("ping_"+std::to_string(ping_num_), "map", ros::Time(0));
        pcl_ros::transformPointCloud(traj_pings_.at(ping_num_).submap_pcl_, *mbes_i_pcl, transformStamped.transform);
        pcl::toROSMsg(*mbes_i_pcl.get(), mbes_i);
        mbes_i.header.frame_id = "ping_" + std::to_string(ping_num_);
        mbes_i.header.stamp = ros::Time::now();
        ping_pub_.publish(mbes_i);
        ROS_INFO("Publishing real mbes");

        *mbes_i_pcl_map = traj_pings_.at(ping_num_).submap_pcl_;
        pcl::toROSMsg(*mbes_i_pcl_map.get(), mbes_i_map);
        mbes_i_map.header.frame_id = "map";
        mbes_i_map.header.stamp = ros::Time::now();
        test_pub_.publish (mbes_i_map);
        ROS_INFO("Publishing GT mbes");
    }
    catch (tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());
    }
    ping_num_ += 5;
}
