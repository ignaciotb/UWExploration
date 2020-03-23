#include "auv_model/auv_motion_simple.hpp"


AUVMotionModel::AUVMotionModel(std::string node_name, ros::NodeHandle &nh):
    node_name_(node_name), nh_(&nh){

    std::string sim_odom_top, sim_pings_top, throttle_top, thruster_top, inclination_top, mbes_sim_as;
    nh_->param<std::string>("odom_sim", sim_odom_top, "/sim_auv/odom");
    nh_->param<std::string>("world_frame", world_frame_, "world");
    nh_->param<std::string>("map_frame", map_frame_, "map");
    nh_->param<std::string>("odom_frame", odom_frame_, "odom");
    nh_->param<std::string>("base_link", base_frame_, "base_link");
    nh_->param<std::string>("mbes_link", mbes_frame_, "mbes_link");
    nh_->param<std::string>("sim_pings", sim_pings_top, "/sim/mbes_pings");
    nh_->param<std::string>("throttle_cmd", throttle_top, "/throttle");
    nh_->param<std::string>("thruster_cmd", thruster_top, "/thruster");
    nh_->param<std::string>("inclination_cmd", inclination_top, "/inclination");
    nh_->param<std::string>("mbes_sim_as", mbes_sim_as, "mbes_sim_action");

    odom_pub_ = nh_->advertise<nav_msgs::Odometry>(sim_odom_top, 50);
    thruster_sub_ = nh_->subscribe(thruster_top, 1, &AUVMotionModel::thrustCB, this);
    incl_sub_ = nh_->subscribe(inclination_top, 1, &AUVMotionModel::inclinationCB, this);
    throttle_sub_ = nh_->subscribe(throttle_top, 1, &AUVMotionModel::throttleCB, this);
    sim_ping_pub_ = nh_->advertise<sensor_msgs::PointCloud2>(sim_pings_top, 10);

    ac_ = new actionlib::SimpleActionClient<auv_2_ros::MbesSimAction>(mbes_sim_as, true);

    tfListener_ = new tf2_ros::TransformListener(tfBuffer_);
}

AUVMotionModel::~AUVMotionModel(){

}

void AUVMotionModel::thrustCB(const std_msgs::Float64ConstPtr& thrust_msg){
    latest_thrust_ = thrust_msg->data;
}

void AUVMotionModel::throttleCB(const std_msgs::Float64ConstPtr& throttle_msg){
    latest_throttle_ = throttle_msg->data;
}

void AUVMotionModel::inclinationCB(const std_msgs::Float64ConstPtr& inclination_msg){
    latest_inclination_ = inclination_msg->data;
}


void AUVMotionModel::init(){

    time_now_ = ros::Time::now();
    time_prev_ = ros::Time::now();

    //Initialize AUV pose on top of odom frame
    prev_odom_.header.stamp = ros::Time::now();
    prev_odom_.header.frame_id = odom_frame_;
    prev_odom_.child_frame_id = base_frame_;
    prev_odom_.pose.pose.position.x = 0;
    prev_odom_.pose.pose.position.y = 0;
    prev_odom_.pose.pose.position.z = 0;
    prev_odom_.pose.pose.orientation.x = 0;
    prev_odom_.pose.pose.orientation.y = 0;
    prev_odom_.pose.pose.orientation.z = 0;
    prev_odom_.pose.pose.orientation.w = 1;

    while(!ac_->waitForServer(ros::Duration(1.0))  && ros::ok()){
        ROS_INFO_NAMED(node_name_, "Waiting for action server");
    }

    try {
        tflistener_.waitForTransform(odom_frame_, map_frame_, ros::Time(0), ros::Duration(10.0) );
        tflistener_.lookupTransform(odom_frame_, map_frame_, ros::Time(0), tf_odom_map_);
        ROS_INFO("Locked transform map --> odom");
    }
    catch(tf::TransformException &exception) {
        ROS_ERROR("%s", exception.what());
        ros::Duration(1.0).sleep();
    }

    try {
        tflistener_.waitForTransform(mbes_frame_, base_frame_, ros::Time(0), ros::Duration(10.0) );
        tflistener_.lookupTransform(mbes_frame_, base_frame_, ros::Time(0), tf_mbes_base_);
        ROS_INFO("Locked transform base --> sensor");
    }
    catch(tf::TransformException &exception) {
        ROS_ERROR("%s", exception.what());
        ros::Duration(1.0).sleep();
    }

    ROS_INFO("Initialized AUV motion model");
}


void AUVMotionModel::updateMotion(const ros::TimerEvent&){

    time_now_ = ros::Time::now();
    double dt = (time_now_ - time_prev_).toSec();

    tf::Quaternion q_prev;
    tf::quaternionMsgToTF(prev_odom_.pose.pose.orientation, q_prev);
    tf::Matrix3x3 m(q_prev);
    double roll_prev, pitch_prev, yaw_prev;
    m.getRPY(roll_prev, pitch_prev, yaw_prev);
    double roll_now, pitch_now, yaw_now;
    roll_now = 0.0;
    pitch_now = pitch_prev+latest_inclination_*dt;
    yaw_now = yaw_prev+latest_thrust_*dt;

    Eigen::Vector3d vel_t = {latest_throttle_*std::cos(yaw_now)*std::cos(pitch_now),
                             latest_throttle_*std::sin(yaw_now)*std::cos(pitch_now),
                             -latest_throttle_*std::sin(pitch_now)};

    // TODO: find a safer way to do this
    latest_throttle_ = 0;
    latest_thrust_ = 0;
    latest_inclination_ = 0;

    // Publish odom
    nav_msgs::Odometry odom;
    odom.header.stamp = time_now_;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id = base_frame_;
    odom.pose.pose.position.x = prev_odom_.pose.pose.position.x + vel_t[0]*dt;
    odom.pose.pose.position.y = prev_odom_.pose.pose.position.y + vel_t[1]*dt;
    odom.pose.pose.position.z = prev_odom_.pose.pose.position.z + vel_t[2]*dt;
    odom.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll_now, pitch_now, yaw_now);

    // And velocities
    tf::Quaternion q_now = tf::createQuaternionFromRPY(roll_now, pitch_now, yaw_now);
    tf::Quaternion q_vel = q_now * q_prev.inverse();
    tf::Matrix3x3 m_vel(q_vel);
    double roll_vel, pitch_vel, yaw_vel;
    m_vel.getRPY(roll_vel, pitch_vel, yaw_vel);

    odom.twist.twist.linear.x = vel_t.x();
    odom.twist.twist.linear.y = vel_t.y();
    odom.twist.twist.linear.y = vel_t.z();
    odom.twist.twist.angular.x = roll_vel/dt;
    odom.twist.twist.angular.y = pitch_vel/dt;
    odom.twist.twist.angular.z = yaw_vel/dt;
    odom_pub_.publish(odom);

    // Broadcast odom ---> base link
    new_base_link_.header.frame_id = odom_frame_;
    new_base_link_.child_frame_id = base_frame_;
    new_base_link_.header.stamp = ros::Time::now();
    new_base_link_.transform.translation.x = odom.pose.pose.position.x;
    new_base_link_.transform.translation.y = odom.pose.pose.position.y;
    new_base_link_.transform.translation.z = odom.pose.pose.position.z;
    new_base_link_.transform.rotation = odom.pose.pose.orientation;
    br_.sendTransform(new_base_link_);

    time_prev_ = time_now_;
    prev_odom_ = odom;

    this->updateMeas();
}

void AUVMotionModel::updateMeas(){

        clock_t tStart = clock();
        // Latest tf mbes-->odom
        tf::Transform tf_mbes_odom;
        tf::transformMsgToTF(new_base_link_.transform, tf_mbes_odom);
        tf_mbes_odom.mult(tf_mbes_base_, tf_mbes_odom.inverse());

        // Latest tf mbes-->map
        tf::Transform tf_mbes_map;
        tf_mbes_map.mult(tf_mbes_odom, tf_odom_map_);
        geometry_msgs::Transform transform;
        tf::transformTFToMsg(tf_mbes_map, transform);

        auv_2_ros::MbesSimGoal mbes_goal;
        mbes_goal.mbes_pose.header.frame_id = mbes_frame_;
        mbes_goal.mbes_pose.child_frame_id = map_frame_;
        mbes_goal.mbes_pose.header.stamp = ros::Time::now();
        mbes_goal.mbes_pose.transform = transform;
        ac_->sendGoal(mbes_goal);

        ac_->waitForResult(ros::Duration(1.0));
        actionlib::SimpleClientGoalState state = ac_->getState();
        if (state == actionlib::SimpleClientGoalState::SUCCEEDED){
            sensor_msgs::PointCloud2 mbes_msg;
            auv_2_ros::MbesSimResult mbes_res = *ac_->getResult();
//            mbes_msg.header.frame_id = mbes_res.sim_mbes.header.frame_id;
//            mbes_msg.header.stamp = mbes_res.sim_mbes.header.stamp;
            mbes_msg = mbes_res.sim_mbes;
            sim_ping_pub_.publish(mbes_msg);
        }
//        printf("AUV Motion time taken: %.4fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
}
