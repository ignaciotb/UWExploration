#include "rbpf_par_slam.h"
#include <ros/callback_queue.h>


int main(int argc, char** argv){
    ros::init(argc, argv, "rbpf_node");

    ros::NodeHandle nh("~");
    ros::NodeHandle nh_mb("~");
    ros::CallbackQueue rbpf_queue;
    ros::CallbackQueue mb_queue;
    nh.setCallbackQueue(&rbpf_queue);
    nh_mb.setCallbackQueue(&mb_queue);

    boost::shared_ptr<RbpfSlam> rbpf(new RbpfSlam(nh, nh_mb));

    // Spinner for AUV interface callbacks
    ros::AsyncSpinner spinner_rbpf(4, &rbpf_queue);
    spinner_rbpf.start();
    
    // Spinner for SVGPs minibatch callbacks
    int pc;
    nh.param<int>(("particle_count"), pc, 10);
    // Option 1
    ros::AsyncSpinner spinner_mb(pc, &mb_queue);
    spinner_mb.start();

    // Option 2
    // ros::MultiThreadedSpinner spinner_mb(pc);
    // spinner_mb.spin(&mb_queue);

    // Option 3
    // while (nh_mb.ok())
    // {
    //     mb_queue.callAvailable();
    // }

    ros::waitForShutdown();
    if(!ros::ok()){
        rbpf.reset();
    }

    return 0;
}