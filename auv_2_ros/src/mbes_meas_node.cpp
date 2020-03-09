#include "auv_2_ros/mbes_meas.hpp"

#include <ros/callback_queue.h>


int main(int argc, char** argv){
    ros::init(argc, argv, "mbes_meas_node");

    ros::NodeHandle nh_nav("~");

    std::string map_str;
    double rate;
    nh_nav.param<std::string>("map_cereal", map_str, "map.cereal");
    nh_nav.param<double>("sim_freq", rate, 1);

    boost::filesystem::path map_path(map_str);
    std::cout << "Map path " << boost::filesystem::basename(map_path) << std::endl;

    ros::CallbackQueue nav_queue;
    nh_nav.setCallbackQueue(&nav_queue);

    MbesMeas *mbes_meas = new MbesMeas(ros::this_node::getName(), nh_nav);
    mbes_meas->init(map_path);
    ros::Timer timer1 = nh_nav.createTimer(ros::Duration(rate), &MbesMeas::broadcastW2MTf, mbes_meas);

    ros::AsyncSpinner spinner_nav(4, &nav_queue);
    spinner_nav.start();
//    ros::spin();

    ros::waitForShutdown();

    if(!ros::ok()){
        delete mbes_meas;
    }

    return 0;
}
