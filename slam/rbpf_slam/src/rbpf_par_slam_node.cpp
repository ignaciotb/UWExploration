#include "rbpf_par_slam.h"
#include <ros/callback_queue.h>


int main(int argc, char** argv){
    ros::init(argc, argv, "rbpf_node");

    ros::NodeHandle nh;
    ros::CallbackQueue nav_queue;
    nh.setCallbackQueue(&nav_queue);

    boost::shared_ptr<RbpfSlam> rbpf(new RbpfSlam(ros::this_node::getName(), nh));

    ros::AsyncSpinner spinner_nav(1, &nav_queue);
    spinner_nav.start();

    ros::waitForShutdown();

    if(!ros::ok()){
        rbpf.reset();
    }

    return 0;
}