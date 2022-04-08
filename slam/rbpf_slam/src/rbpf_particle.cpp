#include "rbpf_particle.h"

sensor_msgs::PointCloud2 pack_cloud(string frame, std::vector<Eigen::RowVector3f> mbes)
{
    typedef pcl::PointCloud<pcl::PointXYZ> PCloud;
    sensor_msgs::PointCloud2 mbes_pcloud; 

    PCloud::Ptr pcl_pcloud(new PCloud);
    pcl_pcloud->header.frame_id = frame;
    pcl_pcloud->height = pcl_pcloud->width = 1;
    for(int i = 0; i < mbes.size(); i++)
    {
        pcl_pcloud->points.push_back(pcl::PointXYZ(mbes[i][0], mbes[i][1], mbes[i][2]));
    }

    pcl::toROSMsg(*pcl_pcloud.get(), mbes_pcloud);
    
    return mbes_pcloud;
}

std::vector<geometry_msgs::Point32> pcloud2ranges_full(sensor_msgs::PointCloud2 point_cloud)
{
    int pc_size = point_cloud.size()
    Eigen::MatrixXXf beams;

    sensor_msgs::PointCloud out_cloud;
    sensor_msgs::convertPointCloud2ToPointCloud(point_cloud, out_cloud);

    for (int i = 0; i < out_cloud.points.size(); ++i)
    {
        Eigen::RowVector3f point;

        point.x = out_cloud.points[i].x;
        point.y = out_cloud.points[i].y;
        point.z = out_cloud.points[i].z;

        beams.push_back(point);
    }

    return beams;
}

// A function to generate numpy linspace 
std::vector<int> linspace(float start, float end, float num)
{
    std::vector<int> linspaced;

    if (0 != num)
    {
        if (1 == num) 
        {
            linspaced.push_back(static_cast<int>(start));
        }
        else
        {
            float delta = (end - start) / (num - 1);

            for (auto i = 0; i < (num - 1); ++i)
            {
                linspaced.push_back(static_cast<int>(start + delta * i));
            }
            // ensure that start and end are exactly the same as the input
            linspaced.push_back(static_cast<int>(end));
        }
    }
    return linspaced;
}