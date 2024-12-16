#include "sss_particle_filter/sss_pf_localization.h"
#include <ros/callback_queue.h>


int main(int argc, char** argv){
    ros::init(argc, argv, "pf_node");

    ros::NodeHandle nh("~");
    ros::NodeHandle nh_mb("~");
    ros::CallbackQueue pf_queue;
    ros::CallbackQueue mb_queue;
    nh.setCallbackQueue(&pf_queue);
    nh_mb.setCallbackQueue(&mb_queue);

    boost::shared_ptr<pfLocalization> pf(new pfLocalization(nh, nh_mb));
    // pf->setup_svgps();

    // Spinner for AUV interface callbacks
    ros::AsyncSpinner spinner_pf(1000, &pf_queue);
    spinner_pf.start();
    
    // Spinner for SVGPs minibatch callbacks
    int pc;
    nh.param<int>(("particle_count"), pc, 10);
    // Option 1
    ros::AsyncSpinner spinner_mb(1000, &mb_queue);
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
        pf.reset();
    }

    return 0;
}