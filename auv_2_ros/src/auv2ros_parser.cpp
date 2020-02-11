#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <cereal/archives/binary.hpp>

#include "data_tools/std_data.h"

#include "submaps_tools/cxxopts.hpp"
#include "submaps_tools/submaps.hpp"
#include "registration/utils_visualization.hpp"

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

// #include <sensor_msgs/PointCloud.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl/PCLHeader.h>

using namespace Eigen;
using namespace std;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudpub;

int main(int argc, char** argv){

    // Initialize ROS node
    ros::init(argc, argv, "point_cloud");
    ros::NodeHandle n;
    // Initialize PointCloud2 publisher
    ros::Publisher pc_pub = n.advertise<sensor_msgs::PointCloud2> ("point_cloud_topic", 4);
    // Create a container for PointCloud2 msg
    sensor_msgs::PointCloud2 outcloud;

    // Inputs
    std::string track_str, map_str, output_str, original, simulation;
    cxxopts::Options options("MyProgram", "One line description of MyProgram");
    options.add_options()
        ("help", "Print help")
        ("output_cereal", "Output graph cereal", cxxopts::value(output_str))
        ("original", "Disturb original trajectory", cxxopts::value(original))
        ("simulation", "Simulation data from Gazebo", cxxopts::value(simulation))
        ("trajectory", "Input AUV GT data", cxxopts::value(track_str))
        ("map", "Localization map", cxxopts::value(map_str));

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        cout << options.help({ "", "Group" }) << endl;
        exit(0);
    }
    if(output_str.empty()){
        output_str = "output_cereal.cereal";
    }

    // Parse input data from cereal files
    SubmapsVec maps_gt, traj_pings;
    MapObj map_loc;
    Eigen::Isometry3d map_tf;
    boost::filesystem::path map_path(map_str);
    boost::filesystem::path auv_path(track_str);
    std::cout << "Map path " << boost::filesystem::basename(map_path) << std::endl;
    std::cout << "AUV path " << boost::filesystem::basename(auv_path) << std::endl;

    if(simulation == "yes"){
        maps_gt = readSubmapsInDir(map_path.string());
    }
    else{
        if(original == "yes"){
            std_data::pt_submaps ss = std_data::read_data<std_data::pt_submaps>(map_path);
            std::tie(map_loc, map_tf)= parseMapAUVlib(ss);
            maps_gt.push_back(map_loc);

            std_data::mbes_ping::PingsT std_pings = std_data::read_data<std_data::mbes_ping::PingsT>(auv_path);
            std::cout << "Number of pings " << std_pings.size() << std::endl;
            traj_pings = parsePingsAUVlib(std_pings, map_tf);
        }
        else{
            std::ifstream is(boost::filesystem::basename(map_path) + ".cereal", std::ifstream::binary);
            {
              cereal::BinaryInputArchive iarchive(is);
              iarchive(maps_gt);
            }
        }
        // Filtering of maps
        PointCloudT::Ptr cloud_ptr (new PointCloudT);
        pcl::UniformSampling<PointT> us_filter;
        us_filter.setInputCloud (cloud_ptr);
        us_filter.setRadiusSearch(2);   // Tune for speed/map resolution
        for(SubmapObj& submap_i: maps_gt){
            *cloud_ptr = submap_i.submap_pcl_;
            us_filter.setInputCloud(cloud_ptr);
            us_filter.filter(*cloud_ptr);
            submap_i.submap_pcl_ = *cloud_ptr;
        }

        // Convert point cloud to message for RVIZ viewing
        pcl::toROSMsg(*cloud_ptr.get(),outcloud);

        // Assign header as map to visualize in RVIZ        
        outcloud.header.frame_id = "map";

        // Publish point cloud message
        ros::Rate loop_rate(.1);
        while (n.ok()){
            pc_pub.publish (outcloud);
            loop_rate.sleep ();
            }
    }
    // Visualization on PCL
    // bool vis = true;
    bool vis = false;
    if (vis){
        PCLVisualizer viewer ("Submaps viewer");
        viewer.loadCameraParameters("Antarctica7");
        SubmapsVisualizer* visualizer = new SubmapsVisualizer(viewer);
        visualizer->setVisualizer(maps_gt, 1);
        while(!viewer.wasStopped ()){
            viewer.spinOnce ();
        }
        viewer.resetStoppedFlag();
    }
    return 0;
}
