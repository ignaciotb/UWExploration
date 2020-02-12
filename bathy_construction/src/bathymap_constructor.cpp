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

#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <geometry_msgs/TransformStamped.h>

using namespace Eigen;
using namespace std;

int main(int argc, char** argv){

    // Initialize ROS node
    ros::init(argc, argv, "point_cloud");
    ros::NodeHandle n;
    ros::Publisher ping_pub = n.advertise<sensor_msgs::PointCloud2> ("mbes_pings", 4);
    ros::Publisher test_pub = n.advertise<sensor_msgs::PointCloud2> ("debug", 4);
    ros::Publisher map_pub = n.advertise<sensor_msgs::PointCloud2> ("map", 4);

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
    SubmapsVec maps_gt, traj_pings;
    MapObj map_loc;
    Eigen::Isometry3d map_tf;
    boost::filesystem::path map_path(map_str);
    boost::filesystem::path auv_path(track_str);
    std::cout << "Map path " << boost::filesystem::basename(map_path) << std::endl;
    std::cout << "AUV path " << boost::filesystem::basename(auv_path) << std::endl;

    // Read map
    std_data::pt_submaps ss = std_data::read_data<std_data::pt_submaps>(map_path);
    std::tie(map_loc, map_tf)= parseMapAUVlib(ss);
    maps_gt.push_back(map_loc);

    // Read pings
    std_data::mbes_ping::PingsT std_pings = std_data::read_data<std_data::mbes_ping::PingsT>(auv_path);
    std::cout << "Number of pings " << std_pings.size() << std::endl;
    traj_pings = parsePingsAUVlib(std_pings, map_tf);

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

    // Publish world-->map-->odom frames
    static tf2_ros::StaticTransformBroadcaster static_broadcaster;
    geometry_msgs::TransformStamped w2m_static_tfStamped, m2o_static_tfStamped;

    w2m_static_tfStamped.header.stamp = ros::Time::now();
    w2m_static_tfStamped.header.frame_id = "world";
    w2m_static_tfStamped.child_frame_id = "map";
    w2m_static_tfStamped.transform.translation.x = map_loc.submap_tf_.translation()[0];
    w2m_static_tfStamped.transform.translation.y = map_loc.submap_tf_.translation()[1];
    w2m_static_tfStamped.transform.translation.z = map_loc.submap_tf_.translation()[2];
    tf2::Quaternion quatw2m;
    Eigen::Vector3d euler = map_loc.submap_tf_.linear().matrix().eulerAngles(0, 1, 2).cast<double>();
    quatw2m.setRPY(euler[0], euler[1], euler[2]);
    w2m_static_tfStamped.transform.rotation.x = quatw2m.x();
    w2m_static_tfStamped.transform.rotation.y = quatw2m.y();
    w2m_static_tfStamped.transform.rotation.z = quatw2m.z();
    w2m_static_tfStamped.transform.rotation.w = quatw2m.w();
    static_broadcaster.sendTransform(w2m_static_tfStamped);

    Eigen::Vector3f map_odom = traj_pings.at(0).submap_tf_.translation();

    m2o_static_tfStamped.header.stamp = ros::Time::now();
    m2o_static_tfStamped.header.frame_id = "map";
    m2o_static_tfStamped.child_frame_id = "odom";
    m2o_static_tfStamped.transform.translation.x = map_odom[0];
    m2o_static_tfStamped.transform.translation.y = map_odom[1];
    m2o_static_tfStamped.transform.translation.z = map_odom[2];
    euler = traj_pings.at(0).submap_tf_.linear().matrix().eulerAngles(0, 1, 2).cast<double>();
    tf2::Quaternion quatm2o;
    quatm2o.setRPY(euler[0], euler[1], euler[2]);
    m2o_static_tfStamped.transform.rotation.x = quatm2o.x();
    m2o_static_tfStamped.transform.rotation.y = quatm2o.y();
    m2o_static_tfStamped.transform.rotation.z = quatm2o.z();
    m2o_static_tfStamped.transform.rotation.w = quatm2o.w();
    static_broadcaster.sendTransform(m2o_static_tfStamped);

    // Publish MBES pings
    sensor_msgs::PointCloud2 mbes_i, mbes_i_map, map;
    PointCloudT::Ptr mbes_i_pcl(new PointCloudT);
    PointCloudT::Ptr mbes_i_pcl_map(new PointCloudT);

    static tf2_ros::TransformBroadcaster br;
    geometry_msgs::TransformStamped new_pings_tf;

    pcl::toROSMsg(*cloud_ptr.get(), map);
    map.header.frame_id = "map";
    map_pub.publish(map);

    ros::Rate loop_rate(1/5.0);
    int j = 2;
    for(int j=0; j<traj_pings.size(); j=j+5){

        new_pings_tf.header.frame_id = "odom";
        new_pings_tf.child_frame_id = "ping" + std::to_string(j);
        Eigen::Vector3f odom_ping_i = traj_pings.at(j).submap_tf_.translation() - map_odom;
        odom_ping_i = traj_pings.at(0).submap_tf_.linear().matrix().transpose() * odom_ping_i;
        new_pings_tf.transform.translation.x = odom_ping_i[0];
        new_pings_tf.transform.translation.y = odom_ping_i[1];
        new_pings_tf.transform.translation.z = odom_ping_i[2];
        tf2::Quaternion quato2p;
        euler = (traj_pings.at(0).submap_tf_.linear().matrix().transpose() *
                 traj_pings.at(j).submap_tf_.linear().matrix()).eulerAngles(0, 1, 2).cast<double>();
        quato2p.setRPY(euler[0], euler[1], euler[2]);
        new_pings_tf.transform.rotation.x = quato2p.x();
        new_pings_tf.transform.rotation.y = quato2p.y();
        new_pings_tf.transform.rotation.z = quato2p.z();
        new_pings_tf.transform.rotation.w = quato2p.w();

        // Original ping (for debugging)
        *mbes_i_pcl_map = traj_pings.at(j).submap_pcl_;
        pcl::toROSMsg(*mbes_i_pcl_map.get(), mbes_i_map);
        mbes_i_map.header.frame_id = "map";
        test_pub.publish (mbes_i_map);

        // Transform point from map to ping i
        for(PointT& point_i: traj_pings.at(j).submap_pcl_.points){
            Eigen::Vector3f point = (point_i.getVector3fMap() - (map_odom + odom_ping_i));
            point = (traj_pings.at(0).submap_tf_.linear().matrix().transpose() *
                     traj_pings.at(j).submap_tf_.linear().matrix()) * point;
            point_i = PointT(point[0], point[1], point[2]);
        }
        *mbes_i_pcl = traj_pings.at(j).submap_pcl_;
        pcl::toROSMsg(*mbes_i_pcl.get(),mbes_i);
        mbes_i.header.frame_id = "ping" + std::to_string(j);

        if(!ros::ok()){
            std::cout << "Not cool! " << std::endl;
            return 0;
        }

//        while(ros::ok()){
//            mbes_i.header.stamp = ros::Time::now();
            new_pings_tf.header.stamp = ros::Time::now();
            br.sendTransform(new_pings_tf);
            ping_pub.publish (mbes_i);
//            test_pub.publish (mbes_i_map);
//            map_pub.publish(map);
            loop_rate.sleep();
            ros::spinOnce();
//        }
    }

    return 0;
}
