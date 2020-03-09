#include "auv_2_ros/auv_2_ros.hpp"


BathymapConstructor::BathymapConstructor(std::string node_name, ros::NodeHandle &nh):
    node_name_(node_name), nh_(&nh){

    std::string gt_pings_top, debug_pings_top, gt_odom_top, sim_pings_top;
    nh_->param<std::string>("mbes_pings", gt_pings_top, "/gt/mbes_pings");
    nh_->param<std::string>("sim_pings", sim_pings_top, "/sim/mbes");
    nh_->param<std::string>("debug_pings", debug_pings_top, "/debug_pings");
    nh_->param<std::string>("odom_gt", gt_odom_top, "/gt/odom");
    nh_->param<std::string>("world_frame", world_frame_, "world");
    nh_->param<std::string>("map_frame", map_frame_, "map");
    nh_->param<std::string>("odom_frame", odom_frame_, "odom");
    nh_->param<std::string>("base_link", base_frame_, "base_link");
    nh_->param<std::string>("mbes_link", mbes_frame_, "mbes_link");

    ping_pub_ = nh_->advertise<sensor_msgs::PointCloud2>(gt_pings_top, 10);
    sim_ping_pub_ = nh_->advertise<sensor_msgs::PointCloud2>(sim_pings_top, 10);
    test_pub_ = nh_->advertise<sensor_msgs::PointCloud2>(debug_pings_top, 10);
    odom_pub_ = nh_->advertise<nav_msgs::Odometry>(gt_odom_top, 50);

    ac_ = new actionlib::SimpleActionClient<auv_2_ros::MbesSimAction>("mbes_meas_node", true);

    tfListener_ = new tf2_ros::TransformListener(tfBuffer_);
    ping_num_ = 0;

    time_now_ = ros::Time::now();
    time_prev_ = ros::Time::now();
}

BathymapConstructor::~BathymapConstructor(){

}


void BathymapConstructor::init(const boost::filesystem::path auv_path){

    // Read pings
    std_data::mbes_ping::PingsT std_pings = std_data::read_data<std_data::mbes_ping::PingsT>(auv_path);
    std::cout << "Number of pings " << std_pings.size() << std::endl;

    // Get fixed transform world --> map frame
    tf::StampedTransform tf_world_map_;
    try {
        tflistener_.waitForTransform(world_frame_, map_frame_, ros::Time(0), ros::Duration(10.0) );
        tflistener_.lookupTransform(world_frame_, map_frame_, ros::Time(0), tf_world_map_);
        ROS_INFO("Locked transform world --> map");
    }
    catch(tf::TransformException &exception) {
        ROS_ERROR("%s", exception.what());
        ros::Duration(1.0).sleep();
    }

    tf::transformTFToEigen(tf_world_map_, map_tf_);
    traj_pings_ = parsePingsAUVlib(std_pings, map_tf_);

    // Store odom tf frame
    odom_tf_ = traj_pings_.at(0).submap_tf_.cast<double>();

    ac_->waitForServer();

    ROS_INFO("Initialized bathymap constructor");
}

void BathymapConstructor::broadcastTf(const ros::TimerEvent&){

    // Publish map-->odom frames
    geometry_msgs::TransformStamped m2o_static_tfStamped;
    Eigen::Vector3d euler;
    m2o_static_tfStamped.header.stamp = ros::Time::now();
    m2o_static_tfStamped.header.frame_id = map_frame_;
    m2o_static_tfStamped.child_frame_id = odom_frame_;
    m2o_static_tfStamped.transform.translation.x = odom_tf_.translation()[0];
    m2o_static_tfStamped.transform.translation.y = odom_tf_.translation()[1];
    m2o_static_tfStamped.transform.translation.z = odom_tf_.translation()[2];
    euler = odom_tf_.linear().matrix().eulerAngles(0, 1, 2);
    tf2::Quaternion quatm2o;
    quatm2o.setRPY(euler[0], euler[1], euler[2]);
    m2o_static_tfStamped.transform.rotation.x = quatm2o.x();
    m2o_static_tfStamped.transform.rotation.y = quatm2o.y();
    m2o_static_tfStamped.transform.rotation.z = quatm2o.z();
    m2o_static_tfStamped.transform.rotation.w = quatm2o.w();
    static_broadcaster_.sendTransform(m2o_static_tfStamped);

    new_base_link_.header.frame_id = odom_frame_;
    new_base_link_.child_frame_id = base_frame_;
    new_base_link_.header.stamp = ros::Time::now();
    Eigen::Vector3d odom_ping_i = (odom_tf_.inverse() *
                                   traj_pings_.at(ping_num_).submap_tf_.translation().cast<double>());
    new_base_link_.transform.translation.x = odom_ping_i[0];
    new_base_link_.transform.translation.y = odom_ping_i[1];
    new_base_link_.transform.translation.z = odom_ping_i[2];
    tf2::Quaternion quato2p;
    euler = (traj_pings_.at(ping_num_).submap_tf_.linear().matrix().cast<double>() *
             odom_tf_.linear().matrix().inverse()).eulerAngles(0, 1, 2);
    quato2p.setRPY(euler[0], euler[1], euler[2]);
    new_base_link_.transform.rotation.x = quato2p.x();
    new_base_link_.transform.rotation.y = quato2p.y();
    new_base_link_.transform.rotation.z = quato2p.z();
    new_base_link_.transform.rotation.w = quato2p.w();
    br_.sendTransform(new_base_link_);

    this->publishOdom(odom_ping_i, euler);

    this->run();
}

void BathymapConstructor::publishOdom(Eigen::Vector3d odom_ping_i, Eigen::Vector3d euler){

    // Publish odom
    time_now_ = ros::Time::now();
    nav_msgs::Odometry odom;
    odom.header.stamp = time_now_;
    odom.header.frame_id = odom_frame_;

    odom.pose.pose.position.x = odom_ping_i[0];
    odom.pose.pose.position.y = odom_ping_i[1];
    odom.pose.pose.position.z = odom_ping_i[2];
    odom.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(euler[0], euler[1], euler[2]);
    odom.child_frame_id = base_frame_;

    // Compute linear velocities
    double dt = (time_now_ - time_prev_).toSec();
    Eigen::Vector3d pos_now = Eigen::Vector3d(new_base_link_.transform.translation.x,
                                              new_base_link_.transform.translation.y,
                                              new_base_link_.transform.translation.z);
    Eigen::Vector3d pos_prev = Eigen::Vector3d(prev_base_link_.transform.translation.x,
                                               prev_base_link_.transform.translation.y,
                                               prev_base_link_.transform.translation.z);
    Eigen::Vector3d vel_t = (pos_now - pos_prev)/dt;

    // And angular vels
    tf::Quaternion q_prev;
    Eigen::Quaterniond q_prev_eigen;
    tf::quaternionMsgToTF(prev_base_link_.transform.rotation, q_prev);
    tf::quaternionTFToEigen(q_prev, q_prev_eigen);

    Eigen::Matrix3d rot_now = (traj_pings_.at(ping_num_).submap_tf_.linear().matrix().cast<double>() *
                               odom_tf_.linear().matrix().inverse());
    Eigen::Quaterniond ang_vel = Eigen::Quaterniond(rot_now) * Eigen::Quaterniond(q_prev_eigen.matrix().inverse());
    tf::Quaternion ang_vel_tf;
    tf::quaternionEigenToTF(ang_vel, ang_vel_tf);
    odom.twist.twist.linear.x = vel_t.x();
    odom.twist.twist.linear.y = vel_t.y();
    odom.twist.twist.linear.y = vel_t.z();
    odom.twist.twist.angular.x = ang_vel_tf.x()/dt;
    odom.twist.twist.angular.y = ang_vel_tf.y()/dt;
    odom.twist.twist.angular.z = ang_vel_tf.z()/dt;
    odom_pub_.publish(odom);

    prev_base_link_ = new_base_link_;
    time_prev_ = time_now_;

}

void BathymapConstructor::run(){

    // Publish MBES pings
    sensor_msgs::PointCloud2 mbes_i, mbes_i_map;
    PointCloudT::Ptr mbes_i_pcl(new PointCloudT);
    PointCloudT::Ptr mbes_i_pcl_map(new PointCloudT);

    geometry_msgs::TransformStamped transformStamped;
    try{
        // Transform points from map to ping i
        if(tfBuffer_.canTransform(mbes_frame_, map_frame_, ros::Time(0), ros::Duration(1))){
            transformStamped = tfBuffer_.lookupTransform(mbes_frame_, map_frame_, ros::Time(0));
            pcl_ros::transformPointCloud(traj_pings_.at(ping_num_).submap_pcl_, *mbes_i_pcl, transformStamped.transform);
            pcl::toROSMsg(*mbes_i_pcl.get(), mbes_i);
            mbes_i.header.frame_id = mbes_frame_;
            mbes_i.header.stamp = ros::Time::now();
            ping_pub_.publish(mbes_i);

            // For testing of the action server
            auv_2_ros::MbesSimGoal mbes_goal;
            mbes_goal.mbes_pose = transformStamped;
            ac_->sendGoal(mbes_goal);

            ac_->waitForResult(ros::Duration(1.0));
            actionlib::SimpleClientGoalState state = ac_->getState();
            if (std::strncmp(state.toString().c_str(), "SUCCEEDED", 9) == 0){
                sim_ping_pub_.publish(*ac_->getResult());
            }
            else{
                ROS_WARN("Action %s", state.toString().c_str());
            }
        }

        // Original ping (for debugging)
        *mbes_i_pcl_map = traj_pings_.at(ping_num_).submap_pcl_;
        pcl::toROSMsg(*mbes_i_pcl_map.get(), mbes_i_map);
        mbes_i_map.header.frame_id = map_frame_;
        mbes_i_map.header.stamp = ros::Time::now();
        test_pub_.publish (mbes_i_map);
    }
    catch (tf2::TransformException &ex) {
      ROS_WARN("%s",ex.what());
    }
    ping_num_ += 1;
}