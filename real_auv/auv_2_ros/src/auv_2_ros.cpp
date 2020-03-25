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

//    ac_ = new actionlib::SimpleActionClient<auv_2_ros::MbesSimAction>("mbes_meas_node", true);

    ping_cnt_ = 0;

    time_now_ = ros::Time::now();
    time_prev_ = ros::Time::now();
}

BathymapConstructor::~BathymapConstructor(){

}


void BathymapConstructor::init(const boost::filesystem::path auv_path,
                               const boost::filesystem::path map_path){

    // Read map
    MapObj map_loc;
    std_data::pt_submaps ss = std_data::read_data<std_data::pt_submaps>(map_path);
    std::tie(map_loc, map_tf_)= parseMapAUVlib(ss);

    // Read pings
    std_data::mbes_ping::PingsT std_pings = std_data::read_data<std_data::mbes_ping::PingsT>(auv_path);
    ping_total_ = std_pings.size();
    std::cout << "Number of pings in survey " << ping_total_ << std::endl;
    traj_pings_ = parsePingsAUVlib(std_pings, map_tf_);

    // Store world --> map tf
    world_map_tfmsg_.header.frame_id = world_frame_;
    world_map_tfmsg_.child_frame_id = map_frame_;
    world_map_tfmsg_.transform.translation.x = map_tf_.translation()[0];
    world_map_tfmsg_.transform.translation.y = map_tf_.translation()[1];
    world_map_tfmsg_.transform.translation.z = map_tf_.translation()[2];
    tf2::Quaternion quatw2m;
    Eigen::Vector3d euler = map_tf_.linear().matrix().eulerAngles(0, 1, 2);
    quatw2m.setRPY(euler[0], euler[1], euler[2]);
    quatw2m.normalize();
    world_map_tfmsg_.transform.rotation.x = quatw2m.x();
    world_map_tfmsg_.transform.rotation.y = quatw2m.y();
    world_map_tfmsg_.transform.rotation.z = quatw2m.z();
    world_map_tfmsg_.transform.rotation.w = quatw2m.w();

    // Store map --> odom tf
    odom_tf_ = traj_pings_.at(0).submap_tf_.cast<double>();

    map_odom_tfmsg_.header.frame_id = map_frame_;
    map_odom_tfmsg_.child_frame_id = odom_frame_;
    map_odom_tfmsg_.transform.translation.x = odom_tf_.translation()[0];
    map_odom_tfmsg_.transform.translation.y = odom_tf_.translation()[1];
    map_odom_tfmsg_.transform.translation.z = odom_tf_.translation()[2];
    euler = odom_tf_.linear().matrix().eulerAngles(0, 1, 2);
    tf::Quaternion quatm2o;
    quatm2o.setRPY(euler[0], euler[1], euler[2]);
    quatm2o.normalize();
    map_odom_tfmsg_.transform.rotation.x = quatm2o.x();
    map_odom_tfmsg_.transform.rotation.y = quatm2o.y();
    map_odom_tfmsg_.transform.rotation.z = quatm2o.z();
    map_odom_tfmsg_.transform.rotation.w = quatm2o.w();

    std::cout << "Map to odom tf " << std::endl;
    std::cout << odom_tf_.translation().transpose() << std::endl;
    std::cout << euler.transpose() << std::endl;

    tf::Transform tf_map_odom;
    tf::transformMsgToTF(map_odom_tfmsg_.transform, tf_map_odom);
    odom_map_tf_ = tf_map_odom.inverse();

    try {
        tflistener_.waitForTransform(mbes_frame_, base_frame_, ros::Time(0), ros::Duration(10.0) );
        tflistener_.lookupTransform(mbes_frame_, base_frame_, ros::Time(0), tf_mbes_base_);
        ROS_INFO("Locked transform base --> sensor");
    }
    catch(tf::TransformException &exception) {
        ROS_ERROR("%s", exception.what());
        ros::Duration(1.0).sleep();
    }

    //    ac_->waitForServer();

    ROS_INFO("Initialized auv_2_ros");
}

void BathymapConstructor::broadcastTf(const ros::TimerEvent&){

    // BR world-->map frames
    world_map_tfmsg_.header.stamp = ros::Time::now();
    static_broadcaster_.sendTransform(world_map_tfmsg_);

    // BR map-->odom frames
    map_odom_tfmsg_.header.stamp = ros::Time::now();
    static_broadcaster_.sendTransform(map_odom_tfmsg_);

    // BR odom-->base frames
    new_base_link_.header.frame_id = odom_frame_;
    new_base_link_.child_frame_id = base_frame_;
    new_base_link_.header.stamp = ros::Time::now();
    Eigen::Vector3d odom_ping_i = (odom_tf_.inverse() *
                                   traj_pings_.at(ping_cnt_).submap_tf_.translation().cast<double>());
    new_base_link_.transform.translation.x = odom_ping_i[0];
    new_base_link_.transform.translation.y = odom_ping_i[1];
    new_base_link_.transform.translation.z = odom_ping_i[2];
    tf::Quaternion quato2p;
    Eigen::Vector3d euler = (traj_pings_.at(ping_cnt_).submap_tf_.linear().matrix().cast<double>() *
             odom_tf_.linear().matrix().inverse()).eulerAngles(0, 1, 2);
    quato2p.setRPY(euler[0], euler[1], euler[2]);
    quato2p.normalize();
    new_base_link_.transform.rotation.x = quato2p.x();
    new_base_link_.transform.rotation.y = quato2p.y();
    new_base_link_.transform.rotation.z = quato2p.z();
    new_base_link_.transform.rotation.w = quato2p.w();
    static_broadcaster_.sendTransform(new_base_link_);

    this->publishOdom(odom_ping_i, euler);

    this->publishMeas(ping_cnt_);

    if(ping_cnt_ < ping_total_-1){
        ping_cnt_ += 1;
    }
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
    Eigen::Vector3d pos_now, pos_prev;
    tf::vectorMsgToEigen(new_base_link_.transform.translation, pos_now);
    tf::vectorMsgToEigen(prev_base_link_.transform.translation, pos_prev);
    Eigen::Vector3d vel_t = (pos_now - pos_prev)/dt;

    // And angular vels
    tf::Quaternion q_prev;
    tf::quaternionMsgToTF(prev_base_link_.transform.rotation, q_prev);
    tf::Quaternion q_now = tf::createQuaternionFromRPY(euler[0], euler[1], euler[2]);
    tf::Quaternion q_vel = q_now.normalized() * q_prev.inverse().normalized();

    tf::Matrix3x3 m_vel(q_vel);
    double roll_vel, pitch_vel, yaw_vel;
    m_vel.getRPY(roll_vel, pitch_vel, yaw_vel);

    odom.twist.twist.linear.x = vel_t.x();
    odom.twist.twist.linear.y = vel_t.y();
    odom.twist.twist.linear.z = vel_t.z();
    odom.twist.twist.angular.x = roll_vel/dt;
    odom.twist.twist.angular.y = pitch_vel/dt;
    odom.twist.twist.angular.z = yaw_vel/dt;
    odom_pub_.publish(odom);

    prev_base_link_ = new_base_link_;
    time_prev_ = time_now_;

}

void BathymapConstructor::publishMeas(int ping_num){

    // Publish MBES pings
    sensor_msgs::PointCloud2 mbes_i, mbes_i_map;
    PointCloudT::Ptr mbes_i_pcl(new PointCloudT);
    PointCloudT::Ptr mbes_i_pcl_map(new PointCloudT);

    tf::Transform tf_mbes_odom;
    tf::transformMsgToTF(new_base_link_.transform, tf_mbes_odom);
    tf_mbes_odom.mult(tf_mbes_base_, tf_mbes_odom.inverse());

    // Latest tf mbes-->map
    tf::Transform tf_mbes_map;
    tf_mbes_map.mult(tf_mbes_odom, odom_map_tf_);

    pcl_ros::transformPointCloud(traj_pings_.at(ping_num).submap_pcl_, *mbes_i_pcl, tf_mbes_map);
    pcl::toROSMsg(*mbes_i_pcl.get(), mbes_i);
    mbes_i.header.frame_id = mbes_frame_;
    mbes_i.header.stamp = ros::Time::now();
    ping_pub_.publish(mbes_i);

//            // For testing of the action server
//            auv_2_ros::MbesSimGoal mbes_goal;
//            mbes_goal.mbes_pose = transformStamped;
//            ac_->sendGoal(mbes_goal);

//            bool finished = ac_->waitForResult(ros::Duration(10.0));
//            if (finished){
//                actionlib::SimpleClientGoalState state = ac_->getState();
//                if (state == actionlib::SimpleClientGoalState::SUCCEEDED){
//                    sensor_msgs::PointCloud2 mbes_msg;
//                    auv_2_ros::MbesSimResult mbes_res = *ac_->getResult();
//                    mbes_msg = mbes_res.sim_mbes;
//                    sim_ping_pub_.publish(mbes_msg);
//                }
//                else{
//                    ROS_WARN("Action %s", state.toString().c_str());
//                }
//            }
//        }

    // Original ping (for debugging)
    *mbes_i_pcl_map = traj_pings_.at(ping_num).submap_pcl_;
    pcl::toROSMsg(*mbes_i_pcl_map.get(), mbes_i_map);
    mbes_i_map.header.frame_id = map_frame_;
    mbes_i_map.header.stamp = ros::Time::now();
    test_pub_.publish (mbes_i_map);
}
