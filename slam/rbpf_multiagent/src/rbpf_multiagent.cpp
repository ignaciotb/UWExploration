#include <rbpf_multiagent/rbpf_multiagent.hpp>

RbpfMultiagent::RbpfMultiagent(ros::NodeHandle &nh, ros::NodeHandle &nh_mb): RbpfSlam(nh, nh_mb){

    // The mission waypoints as a path
    std::string area_topic;
    nh_->param<string>(("area_topic"), area_topic, "/waypoints");
    area_sub_ = nh_->subscribe(area_topic, 1, &RbpfMultiagent::area_cb, this);
}

// TODO: connect to topic with survey area boundaries
void RbpfMultiagent::area_cb(const nav_msgs::PathConstPtr& wp_path)
{
    if (wp_path->poses.size() > 0)
    {
        // This service will start the auv simulation or auv_2_ros nodes to start the mission
        nh_->param<string>(("synch_topic"), synch_top_, "/pf_synch");
        // srv_server_ = nh_->advertiseService(synch_top_, &RbpfMultiagent::empty_srv, this);
        srv_server_ = nh_->advertiseService(synch_top_, &RbpfSlam::empty_srv, (RbpfSlam*) this);
    }
    else{
        ROS_WARN("Received empty mission");
    }

}
