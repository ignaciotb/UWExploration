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
// Include for publishing point cloud and pose data
#include <ros/ros.h>
// Include specifically for point cloud
#include <pcl_ros/point_cloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
// Include specifically for poses
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/Pose.h>
#include <tf2/LinearMath/Quaternion.h>

using namespace Eigen;
using namespace std;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudpub;

// Function for unpacking ping poses into PoseArray msg
// Based on parsePingsAUVlib from submaps.cpp
geometry_msgs::PoseArray parsePingsPoseArray(std_data::mbes_ping::PingsT& pings, const Eigen::Isometry3d& map_tf){
    // Define PoseArray msg
    geometry_msgs::PoseArray pings_poses;
    std::cout << std::fixed;
    std::cout << std::setprecision(10);

    // For every .all file
    Isometry3d submap_tf;
    for (auto pos = pings.begin(); pos != pings.end(); ) {
        auto next = std::find_if(pos, pings.end(), [&](const std_data::mbes_ping& ping) {
            return ping.first_in_file_ && (&ping != &(*pos));
        });
        std_data::mbes_ping::PingsT track_pings;
        track_pings.insert(track_pings.end(), pos, next);
        if (pos == next) {
            break;
        }

        // get the direction of the submap as the mean direction
        Vector3d ang;
        Vector3d pose;

        // For every ping in the .all file
        for (std_data::mbes_ping& ping : track_pings) {
            // Define Pose msg for every pose
            geometry_msgs::Pose ping_pose;
            // Quaternion for pose orientation
            tf2::Quaternion quat;

            for (Vector3d& p : ping.beams) {
                p -= map_tf.translation();
            }
            Vector3d dir = ping.beams.back() - ping.beams.front();

            ang << ping.roll_, ping.pitch_, 0;
            ang *= M_PI/180.0;
            ang[2] = std::atan2(dir(1), dir(0)) + M_PI/2.0;
            // Calculate pose and poss into pose_msg
            pose = (ping.pos_- map_tf.translation());
            ping_pose.position.x = pose(0);
            ping_pose.position.y = pose(1);
            ping_pose.position.z = pose(2);

            // Build quaternion from angle and pass to pose_msg
            quat.setRPY(ang(0), ang(1), ang(2));
            ping_pose.orientation.x = quat[0];
            ping_pose.orientation.y = quat[1];
            ping_pose.orientation.z = quat[2];
            ping_pose.orientation.w = quat[3];
            // Add pose to PoseArray
            pings_poses.poses.push_back(ping_pose);
        }
        pos = next;
    }
    //return pings_subs;
    return pings_poses;
}

int main(int argc, char** argv){

    // Initialize ROS node
    ros::init(argc, argv, "point_cloud");
    ros::NodeHandle n;
    // Initialize PointCloud2 publisher
    ros::Publisher pc_pub = n.advertise<sensor_msgs::PointCloud2> ("point_cloud_topic", 4);
    // Create a container for PointCloud2 msg
    sensor_msgs::PointCloud2 outcloud;

    // Initialize trajectory publisher - posearray
    ros::Publisher traj_pub = n.advertise<geometry_msgs::PoseArray> ("trajectory_topic", 10);

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
    geometry_msgs::PoseArray traj_poses;
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
            // traj_pings = parsePingsAUVlib(std_pings, map_tf);
            traj_poses = parsePingsPoseArray(std_pings, map_tf);
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
        traj_poses.header.frame_id = "map";

        // Publish PoseArray and PointCloud2 messages
        ros::Rate loop_rate(.1);
        while (n.ok()){
            pc_pub.publish (outcloud);
            traj_pub.publish (traj_poses);
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
