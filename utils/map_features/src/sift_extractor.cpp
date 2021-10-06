#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>

#include <iostream>

#include <sensor_msgs/PointCloud2.h>

#include <eigen_conversions/eigen_msg.h>
#include <Eigen/Core>

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudT;
typedef pcl::PointXYZ PointT;

namespace pcl
{
    template <>
    struct SIFTKeypointFieldSelector<PointXYZ>
    {
        inline float
        operator()(const PointXYZ &p) const
        {
            return p.z;
        }
    };
}

void siftExtractionCB(const sensor_msgs::PointCloud2Ptr &map_cloud)
{
    // Parameters for sift computation
    std::cout << "Map received" << std::endl;
    const float min_scale = 0.01f;
    const int n_octaves = 6;
    const int n_scales_per_octave = 4;
    const float min_contrast = 0.005f;

    PointCloudT::Ptr cloud_xyz(new PointCloudT);
    pcl::fromROSMsg(*map_cloud, *cloud_xyz);

    // Estimate the sift interest points using z values from xyz as the Intensity variants
    pcl::SIFTKeypoint<pcl::PointXYZ, pcl::PointWithScale> sift;
    pcl::PointCloud<pcl::PointWithScale> result;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    sift.setSearchMethod(tree);
    sift.setScales(min_scale, n_octaves, n_scales_per_octave);
    sift.setMinimumContrast(min_contrast);
    sift.setInputCloud(cloud_xyz);
    sift.compute(result);

    std::cout << "No of points in original submap " << map_cloud->data.size() << std::endl;
    std::cout << "No of SIFT points in the result are " << result.size() << std::endl;

    PointCloudT::Ptr cloud_temp(new PointCloudT);
    copyPointCloud(result, *cloud_temp);

    // Save
    pcl::io::savePCDFileASCII ("/home/torroba/Downloads/submaps/sift_map.pcd",
                               *cloud_temp);

    // Kill node when done
    ros::shutdown();
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "submap_constructor_node");
    ros::NodeHandle nh("~");

    std::string map_top, sift_path;
    nh.param<std::string>("map_mbes", map_top, "map_top");
    nh.param<std::string>("sift_map_path", sift_path, "sift_path");
    
    ros::Subscriber sub = nh.subscribe(map_top, 1, siftExtractionCB);
    
    ros::spin();

  return 0;
}