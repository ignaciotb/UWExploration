#include "auv_2_ros/auv_2_ros.hpp"


int main(int argc, char** argv){

    ros::init(argc, argv, "auv_2_ros");
    ros::NodeHandle nh("~");

    // Inputs
    std::string track_str, output_str;
    double rate;
    nh.param<std::string>("trajectory_cereal", track_str, "trajectory.cereal");
    nh.param<double>("replay_rate", rate, 1);

    // Parse input data from cereal files
    boost::filesystem::path auv_path(track_str);
    std::cout << "AUV trajectory input " << auv_path << std::endl;

    BathymapConstructor* map_constructor = new BathymapConstructor(ros::this_node::getName(), nh);
    map_constructor->init(auv_path);
    ros::Timer timer1 = nh.createTimer(ros::Duration(rate), &BathymapConstructor::broadcastTf, map_constructor);

    // ros::Duration(10.0).sleep(); // TODO: make it wait for the PF
    ros::spin();
    ros::waitForShutdown();

    if(!ros::ok()){
        delete map_constructor;
    }
    ROS_INFO("AUV_2_ROS finished");

    return 0;
}
