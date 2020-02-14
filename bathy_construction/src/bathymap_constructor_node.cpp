#include "bathy_construction/bathymap_constructor.hpp"


int main(int argc, char** argv){

    ros::init(argc, argv, "bathymap_constructor");
    ros::NodeHandle nh;

    // Inputs
    std::string track_str, map_str, output_str, original, simulation;
    cxxopts::Options options("MyProgram", "One line description of MyProgram");
    options.add_options()
        ("help", "Print help")
        ("trajectory", "Input AUV GT data", cxxopts::value(track_str))
        ("map", "Localization map", cxxopts::value(map_str));

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        cout << options.help({ "", "Group" }) << endl;
        exit(0);
    }

    // Parse input data from cereal files
    boost::filesystem::path map_path(map_str);
    boost::filesystem::path auv_path(track_str);
    std::cout << "Map path " << boost::filesystem::basename(map_path) << std::endl;
    std::cout << "AUV path " << boost::filesystem::basename(auv_path) << std::endl;

    BathymapConstructor* map_constructor = new BathymapConstructor(ros::this_node::getName(), nh);
    map_constructor->init(map_path, auv_path);
    ros::Timer timer1 = nh.createTimer(ros::Duration(1), &BathymapConstructor::broadcastTf, map_constructor);
//    ros::Timer timer2 = nh.createTimer(ros::Duration(5), &BathymapConstructor::run, map_constructor);
    ros::spin();

    ros::waitForShutdown();

    if(!ros::ok()){
        delete map_constructor;
    }
    ROS_INFO("Finishing MBES receptor");

    return 0;
}
