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
    // One thread per particle?
    int pc;
    nh.param<int>(("particle_count"), pc, 10);
    // Spinner for SVGPs minibatch callbacks 
    ros::AsyncSpinner spinner_mb(pc, &mb_queue);

    spinner_rbpf.start();
    spinner_mb.start();

    ros::waitForShutdown();
    if(!ros::ok()){
        rbpf.reset();
    }

    return 0;
}