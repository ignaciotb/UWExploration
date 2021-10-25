#include "auv_2_ros/auv_2_ros.hpp"


BathymapConstructor::BathymapConstructor(std::string node_name, ros::NodeHandle &nh):
    node_name_(node_name), nh_(&nh){

    std::string gt_pings_top, debug_pings_top, mbes_as_name, gt_odom_top,
    sim_pings_top, enable_top;

    // nh_->param<std::string>("mbes_pings", gt_pings_top, "/gt/mbes_pings");
    nh_->param<std::string>("mbes_pings_topic", gt_pings_top, "/gt/mbes");
    nh_->param<std::string>("debug_pings", debug_pings_top, "/debug_pings");
    nh_->param<std::string>("odom_topic", gt_odom_top, "/gt/odom");
    nh_->param<std::string>("world_frame", world_frame_, "world");
    nh_->param<std::string>("map_frame", map_frame_, "map");
    nh_->param<std::string>("odom_frame", odom_frame_, "odom");
    nh_->param<std::string>("base_link", base_frame_, "base_link");
    nh_->param<std::string>("mbes_link", mbes_frame_, "mbes_link");
    nh_->param<std::string>("mini_link", mini_frame_, "mini_link");
    nh_->param<std::string>("survey_finished_top", enable_top, "enable");
    // nh_->param<std::string>("mbes_sim_as", mbes_as_name, "mbes_sim_server");
    nh_->param<std::string>("synch_topic", synch_name_, "/pf/synch");
    nh_->param<bool>("change_detection", change_detection_, false);
    nh_->param<bool>("add_mini", add_mini_, false);
    nh_->param<int>("n_beams_mbes", beams_num_, 100);
    nh_->param<int>("start_mission_ping_num", first_ping_, 0);
    nh_->param<int>("end_mission_ping_num", last_ping_, 0);

    ping_pub_ = nh_->advertise<sensor_msgs::PointCloud2>(gt_pings_top, 100);
    // sim_ping_pub_ = nh_->advertise<sensor_msgs::PointCloud2>(sim_pings_top, 1);
    test_pub_ = nh_->advertise<sensor_msgs::PointCloud2>(debug_pings_top, 1);
    odom_pub_ = nh_->advertise<nav_msgs::Odometry>(gt_odom_top, 100);
    enable_pub_ = nh_->advertise<std_msgs::Bool>(enable_top, 1);
    // pf_synch_sub_ = nh_->subscribe(synch_name, 1, &BathymapConstructor::synchCB, this);
    // ac_ = new actionlib::SimpleActionClient<auv_2_ros::MbesSimAction>(mbes_as_name, true);

    ping_cnt_ = first_ping_;

    time_now_ = ros::Time::now();
    start_replay_ = false;


}

BathymapConstructor::~BathymapConstructor(){

}

// void BathymapConstructor::synchCB(const std_msgs::BoolConstPtr& synch_msg){
//     std::cout << "Synch signal received" << std::endl;
//     start_replay_ = synch_msg->data;
//     time_prev_ = ros::Time::now();

// }

void BathymapConstructor::initMiniFrames(std::vector<Eigen::Vector3d>& minis_poses){

    // Store map --> mini tf
    int cnt = 0;
    geometry_msgs::TransformStamped map_mini_tfmsg;
    Eigen::Isometry3d mini_tf;
    for(Eigen::Vector3d& mini_i: minis_poses){
        mini_tf.translation() = mini_i;//Eigen::Vector3d(15,-30,-15.5);
        Eigen::Quaterniond rot;
        rot.setIdentity();
        mini_tf.linear() = rot.toRotationMatrix();

        map_mini_tfmsg.header.frame_id = map_frame_;
        map_mini_tfmsg.child_frame_id = mini_frame_ + "_" + std::to_string(cnt);
        map_mini_tfmsg.transform.translation.x = mini_tf.translation()[0];
        map_mini_tfmsg.transform.translation.y = mini_tf.translation()[1];
        map_mini_tfmsg.transform.translation.z = mini_tf.translation()[2];
        Eigen::Vector3d euler = mini_tf.linear().matrix().eulerAngles(0, 1, 2);
        tf::Quaternion quatm2m;
        quatm2m.setRPY(euler[0], euler[1], euler[2]);
        quatm2m.normalize();
        map_mini_tfmsg.transform.rotation.x = quatm2m.x();
        map_mini_tfmsg.transform.rotation.y = quatm2m.y();
        map_mini_tfmsg.transform.rotation.z = quatm2m.z();
        map_mini_tfmsg.transform.rotation.w = quatm2m.w();
        cnt ++;
        map_mini_tfmsgs_.push_back(map_mini_tfmsg);
    }
}


void BathymapConstructor::init(const boost::filesystem::path auv_path){

    // MBES to base transform
    try {
        tflistener_.waitForTransform(mbes_frame_, base_frame_, ros::Time(0), ros::Duration(10.0) );
        tflistener_.lookupTransform(mbes_frame_, base_frame_, ros::Time(0), tf_mbes_base_);
        ROS_INFO("Locked transform base --> sensor");
    }
    catch(tf::TransformException &exception) {
        ROS_ERROR("%s", exception.what());
        ros::Duration(1.0).sleep();
    }
    tf::transformTFToEigen(tf_mbes_base_, mbes_base_mat_);
    
    // Read pings
    std_data::mbes_ping::PingsT std_pings = std_data::read_data<std_data::mbes_ping::PingsT>(auv_path);
    ping_total_ = std_pings.size();
    std::cout << "Number of pings in survey " << ping_total_ << std::endl;
    traj_pings_ = parsePingsAUVlib(std_pings);
    
    // World to mbes transform
    world_mbes_tf_ = traj_pings_.at(0).submap_tf_.cast<double>();

    // World to map transform (setting the map where the base started the survey)
    world_map_tf_ = world_mbes_tf_;// * mbes_base_mat_;

    // If the last ping is not set, replay the full mission
    last_ping_= (last_ping_ == 0)? ping_total_-1:last_ping_;

    std::cout << "First ping " << first_ping_ << std::endl;
    std::cout << "Last ping " << last_ping_ << std::endl;

    // Store world --> map tf
    world_map_tfmsg_.header.frame_id = world_frame_;
    world_map_tfmsg_.child_frame_id = map_frame_;
    world_map_tfmsg_.transform.translation.x = world_map_tf_.translation()[0];
    world_map_tfmsg_.transform.translation.y = world_map_tf_.translation()[1];
    world_map_tfmsg_.transform.translation.z = world_map_tf_.translation()[2];
    tf2::Quaternion quatw2m;
    Eigen::Vector3d euler = world_map_tf_.linear().matrix().eulerAngles(0, 1, 2);
    quatw2m.setRPY(euler[0], euler[1], euler[2]);
    quatw2m.normalize();
    world_map_tfmsg_.transform.rotation.x = quatw2m.x();
    world_map_tfmsg_.transform.rotation.y = quatw2m.y();
    world_map_tfmsg_.transform.rotation.z = quatw2m.z();
    world_map_tfmsg_.transform.rotation.w = quatw2m.w();

    // Store map --> minis tfs
    // std::vector<Eigen::Vector3d> minis_poses;
//    minis_poses.push_back(Eigen::Vector3d(15,-30,-15.5));
    // minis_poses.push_back(Eigen::Vector3d(-220,-20,-17));
    // minis_poses.push_back(Eigen::Vector3d(-200,50,-17));
    // initMiniFrames(minis_poses);

    // Store map --> mbes tf
    map_mbes_tf_ = traj_pings_.at(ping_cnt_).submap_tf_.cast<double>();

    // Map to odom transform (setting the odom where the base starts replying the survey)
    map_odom_tf_ = map_mbes_tf_;// * mbes_base_mat_;

    map_odom_tfmsg_.header.frame_id = map_frame_;
    map_odom_tfmsg_.child_frame_id = odom_frame_;
    map_odom_tfmsg_.transform.translation.x = map_odom_tf_.translation()[0];
    map_odom_tfmsg_.transform.translation.y = map_odom_tf_.translation()[1];
    map_odom_tfmsg_.transform.translation.z = map_odom_tf_.translation()[2];
    euler = map_odom_tf_.linear().matrix().eulerAngles(0, 1, 2);
    tf::Quaternion quatm2o;
    quatm2o.setRPY(euler[0], euler[1], euler[2]);
    quatm2o.normalize();
    map_odom_tfmsg_.transform.rotation.x = quatm2o.x();
    map_odom_tfmsg_.transform.rotation.y = quatm2o.y();
    map_odom_tfmsg_.transform.rotation.z = quatm2o.z();
    map_odom_tfmsg_.transform.rotation.w = quatm2o.w();

    // std::cout << "Map to odom tf " << std::endl;
    // std::cout << map_odom_tf_.translation().transpose() << std::endl;
    // std::cout << euler.transpose() << std::endl;

    tf::Transform tf_map_odom;
    tf::transformMsgToTF(map_odom_tfmsg_.transform, tf_map_odom);
    tf_odom_map_ = tf_map_odom.inverse();

    survey_finished_ = false;

    // while(!ac_->waitForServer(ros::Duration(1.0))  && ros::ok()){
    //     ROS_INFO_NAMED(node_name_, "AUV2ROS Waiting for action server");
    // }

    // Set init prev odom--> base transform. Always zero since base starts on top of odom
    prev_base_link_.header.frame_id = odom_frame_;
    prev_base_link_.child_frame_id = base_frame_;
    prev_base_link_.header.stamp = time_now_;                              
    prev_base_link_.transform.translation.x = 0;
    prev_base_link_.transform.translation.y = 0;
    prev_base_link_.transform.translation.z = 0;
    prev_base_link_.transform.rotation.x = 0;
    prev_base_link_.transform.rotation.y = 0;
    prev_base_link_.transform.rotation.z = 0;
    prev_base_link_.transform.rotation.w = 1;
}

void BathymapConstructor::addMiniCar(std::string & mini_name){

    PointCloudT::Ptr cloud_in (new PointCloudT);  // Original point cloud
    mini_name = "/home/torroba18/Downloads/MMT Mini Point Cloud/MMT_Mini_PointCloud.obj";
    if (pcl::io::loadOBJFile(mini_name, *cloud_in) < 0){
        PCL_ERROR ("Error loading cloud \n");
    }
    std::cout << "\nLoaded file " << mini_name << " (" << cloud_in->size () << " points)" << std::endl;

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid(*cloud_in, centroid);
    for(PointT& point_i: cloud_in->points){
        point_i.getArray4fMap() = point_i.getArray4fMap() - centroid.array();
    }

    sensor_msgs::PointCloud2 mbes_i;
    tf::Transform tf_map_mini;
    for(geometry_msgs::TransformStamped& mini_frame: map_mini_tfmsgs_){
        tf::transformMsgToTF(mini_frame.transform, tf_map_mini);

    //    pcl_ros::transformPointCloud(*cloud_in, *cloud_in, tf_map_mini);
        pcl::toROSMsg(*cloud_in.get(), mbes_i);
        mbes_i.header.frame_id = mini_frame.child_frame_id;
        mbes_i.header.stamp = ros::Time::now();
        for(int i=0; i<10; i++){
            ping_pub_.publish(mbes_i);
            ros::Duration(0.1).sleep();
        }
    }
}

void BathymapConstructor::broadcastTf(const ros::TimerEvent&){

    time_now_ = ros::Time::now();

    // BR world-->map frames
    world_map_tfmsg_.header.stamp = time_now_;
    static_broadcaster_.sendTransform(world_map_tfmsg_);

    // BR map-->odom frames
    map_odom_tfmsg_.header.stamp = time_now_;
    static_broadcaster_.sendTransform(map_odom_tfmsg_);

    // BR map-->mini frames
    for(geometry_msgs::TransformStamped& mini_frame: map_mini_tfmsgs_){
        mini_frame.header.stamp = time_now_;
        static_broadcaster_.sendTransform(mini_frame);
    }

    // Don't start survey until PF is up
    // if(start_replay_ != true){
    //     ROS_INFO_NAMED(node_name_, "AUV2ROS Waiting for application to send synch signal");
    //     return;
    // }

    // BR odom-->base at time t
    new_base_link_.header.frame_id = odom_frame_;
    new_base_link_.child_frame_id = base_frame_;
    new_base_link_.header.stamp = time_now_;
    Eigen::Isometry3d odom_base_t = map_odom_tf_.inverse() * 
                                    traj_pings_.at(ping_cnt_).submap_tf_.cast<double>();//*
                                    // mbes_base_mat_;
                                   
    Eigen::Vector3d odom_ping_i = odom_base_t.translation();
    new_base_link_.transform.translation.x = odom_ping_i[0];
    new_base_link_.transform.translation.y = odom_ping_i[1];
    new_base_link_.transform.translation.z = odom_ping_i[2];
    tf::Quaternion quato2p;
    Eigen::Vector3d euler = (odom_base_t.linear().matrix()).eulerAngles(0, 1, 2);
    quato2p.setRPY(euler[0], euler[1], euler[2]);
    quato2p.normalize();
    new_base_link_.transform.rotation.x = quato2p.x();
    new_base_link_.transform.rotation.y = quato2p.y();
    new_base_link_.transform.rotation.z = quato2p.z();
    new_base_link_.transform.rotation.w = quato2p.w();
    static_broadcaster_.sendTransform(new_base_link_);

    // Synch signal to start auv_2_ros survey
    if (start_replay_ == false)
    {
        while (!ros::service::waitForService(synch_name_, -1) && ros::ok())
        {
            std::cout << synch_name_ << std::endl;
            ROS_DEBUG_NAMED(node_name_, "auv_2_ros node waiting for app ");
        }
        ROS_INFO("AUV2ROS UP");
        start_replay_ = true;
    }

    this->publishOdom(odom_ping_i, euler);

    if (ping_cnt_ == first_ping_ && add_mini_)
    {
        std::string mini_name = "/home/torroba/Downloads/MMT Mini Point Cloud/MMT_Mini_PointCloud.obj";
        addMiniCar(mini_name);
    }

//    std::cout << "ping " << ping_cnt_ << std::endl;
    if(ping_cnt_ < last_ping_ && !survey_finished_){
        this->publishMeas(ping_cnt_);
        // if(change_detection_){
        //     this->publishExpectedMeas();
        // }
        // ping_cnt_ += 1;
        ping_cnt_++;
    }
    
    if(ping_cnt_ == last_ping_ && !survey_finished_){
        ROS_INFO_STREAM("Survey finished");
        survey_finished_ = true;
        std_msgs::Bool msg;
        msg.data = true;
        enable_pub_.publish(msg);
        ros::shutdown();
    }

}

void BathymapConstructor::publishOdom(Eigen::Vector3d odom_ping_i, Eigen::Vector3d euler){

    // Publish odom
    nav_msgs::Odometry odom;
    odom.header.stamp = time_now_;
    odom.header.frame_id = odom_frame_;

    odom.pose.pose.position.x = odom_ping_i[0];
    odom.pose.pose.position.y = odom_ping_i[1];
    odom.pose.pose.position.z = odom_ping_i[2];
    odom.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(euler[0], euler[1], euler[2]);
    odom.child_frame_id = base_frame_;

    // Compute linear and angular velocities
    double dt = (time_now_ - time_prev_).toSec();
    Eigen::Vector3d pos_now, pos_prev;
    tf::vectorMsgToEigen(new_base_link_.transform.translation, pos_now);
    tf::vectorMsgToEigen(prev_base_link_.transform.translation, pos_prev);
    Eigen::Vector3d vel_t = (pos_now - pos_prev)/dt;

    tf::Quaternion q_prev;
    tf::quaternionMsgToTF(prev_base_link_.transform.rotation, q_prev);
    tf::Quaternion q_now = tf::createQuaternionFromRPY(euler[0], euler[1], euler[2]);
    tf::Quaternion q_step = q_now.normalized() * q_prev.inverse().normalized();

    tf::Matrix3x3 m_prev(q_prev);
    Eigen::Matrix3d m_prev_e;
    tf::matrixTFToEigen(m_prev, m_prev_e);
    Eigen::Vector3d vel_rel = m_prev_e.inverse() * vel_t;
    tf::Matrix3x3 m_step(q_step);
    double roll_step, pitch_step, yaw_step;
    m_step.getRPY(roll_step, pitch_step, yaw_step);

    odom.twist.twist.linear.x = vel_rel.x();
    odom.twist.twist.linear.y = vel_rel.y();
    odom.twist.twist.linear.z = vel_rel.z();
    odom.twist.twist.angular.x = roll_step / dt;
    odom.twist.twist.angular.y = pitch_step / dt;
    odom.twist.twist.angular.z = yaw_step / dt;
    odom_pub_.publish(odom);

    prev_base_link_ = new_base_link_;
    time_prev_ = time_now_;

}

void BathymapConstructor::publishMeas(int ping_num){

    // Publish MBES pings
    sensor_msgs::PointCloud2 mbes_i, mbes_i_map;
    PointCloudT::Ptr mbes_i_pcl(new PointCloudT);
    PointCloudT::Ptr mbes_i_pcl_filt(new PointCloudT);
    PointCloudT::Ptr mbes_i_pcl_map(new PointCloudT);

    // Transformation map-->mbes
    tf::Transform tf_odom_base;
    tf::transformMsgToTF(new_base_link_.transform, tf_odom_base);
    tf::Transform tf_map_mbes = tf_odom_map_.inverse() * tf_odom_base * tf_mbes_base_.inverse();
    
    // Publish pings in MBES frame
    pcl_ros::transformPointCloud(traj_pings_.at(ping_num).submap_pcl_, *mbes_i_pcl, tf_map_mbes.inverse());

    // Sample down pings to a fix size
    // if (traj_pings_.at(ping_num).submap_pcl_.points.size() > 500){
    //     traj_pings_.at(ping_num).submap_pcl_.points.resize(500);
    //     for(int i=0; i<beams_num_-1; i++){
    //         mbes_i_pcl_filt->points.push_back(traj_pings_.at(ping_num).submap_pcl_.points.at(round((500.0-1.0)*i/(beams_num_-1))));
    //     }
    // //    std::cout << "Ping size after " << mbes_i_pcl_filt->points.size() << std::endl;

    //     std::reverse(mbes_i_pcl_filt->points.begin(), mbes_i_pcl_filt->points.end());
    //     pcl::toROSMsg(*mbes_i_pcl_filt.get(), mbes_i);
    //     mbes_i.header.frame_id = map_frame_;
    //     mbes_i.header.stamp = time_now_;
    //     ping_pub_.publish(mbes_i);
    // }

    std::reverse(mbes_i_pcl->points.begin(), mbes_i_pcl->points.end());
    pcl::toROSMsg(*mbes_i_pcl.get(), mbes_i);
    mbes_i.header.frame_id = mbes_frame_;
    mbes_i.header.stamp = time_now_;
    ping_pub_.publish(mbes_i);

    // Original ping (for debugging)
    *mbes_i_pcl_map = traj_pings_.at(ping_num).submap_pcl_;
    pcl::toROSMsg(*mbes_i_pcl_map.get(), mbes_i_map);
    mbes_i_map.header.frame_id = map_frame_;
    mbes_i_map.header.stamp = time_now_;
    test_pub_.publish (mbes_i_map);
}

// // I don't think this is necessary
// void BathymapConstructor::publishExpectedMeas(){

// //        clock_t tStart = clock();

//         // Transformation map-->mbes
//         tf::Transform tf_odom_base;
//         tf::transformMsgToTF(new_base_link_.transform, tf_odom_base);
//         tf::Transform tf_map_mbes = tf_odom_map_.inverse() * tf_odom_base * tf_mbes_base_.inverse();
//         geometry_msgs::Transform transform_msg;
//         tf::transformTFToMsg(tf_map_mbes, transform_msg);

//         auv_2_ros::MbesSimGoal mbes_goal;
//         mbes_goal.mbes_pose.header.frame_id = map_frame_;
//         mbes_goal.mbes_pose.child_frame_id = mbes_frame_;
//         mbes_goal.mbes_pose.header.stamp = new_base_link_.header.stamp;
//         mbes_goal.mbes_pose.transform = transform_msg;
//         mbes_goal.beams_num.data = beams_num_;
//         ac_->sendGoal(mbes_goal);

//         ac_->waitForResult(ros::Duration(1));
//         actionlib::SimpleClientGoalState state = ac_->getState();
//         if (state == actionlib::SimpleClientGoalState::SUCCEEDED){
//             sensor_msgs::PointCloud2 mbes_msg;
//             auv_2_ros::MbesSimResult mbes_res = *ac_->getResult();
//             mbes_msg = mbes_res.sim_mbes;
//             sim_ping_pub_.publish(mbes_msg);
//         }
//         else{
//             ROS_WARN("Dropped expected meas");
//         }
// //        printf("AUV Motion time taken: %.4fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
// }
