#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_ros/transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <future>
#include <nav_msgs/Odometry.h>

using namespace Eigen;
using namespace std;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudT;

class FixMBESLolo
{

public:
    FixMBESLolo(std::string node_name, ros::NodeHandle &nh) : node_name_(node_name), nh_(&nh)
    {
        // TODO: add params in a launch file
        std::string lolo_ping_top, rbpf_ping_top;
        nh_->param<std::string>("lolo_ping_top", lolo_ping_top, "/lolo/mbes/odom/bathy_points");
        nh_->param<std::string>("rbpf_ping_top", rbpf_ping_top, "/rbpf/mbes_pings");
        nh_->param<std::string>("map_frame", map_frame_, "map");
        nh_->param<std::string>("odom_frame", odom_frame_, "lolo/odom");
        nh_->param<std::string>("base_link", base_frame_, "lolo/base_link");
        nh_->param<std::string>("mbes_link", mbes_frame_, "lolo/mbes_link");
        // MBES to base transform
        tf2_ros::TransformListener tf_listener(tf_buffer_);
        try
        {
            // tf_buffer_.canTransform(odom_frame_, map_frame_, ros::Time(0), ros::Duration(10.));
            auto asynch_1 = std::async(std::launch::async, [this]
                                       { return tf_buffer_.lookupTransform(odom_frame_, map_frame_,
                                                                           ros::Time(0), ros::Duration(30.)); });

            geometry_msgs::TransformStamped tfmsg_odom_map = asynch_1.get();
            // geometry_msgs::TransformStamped tfmsg_odom_map = tf_buffer_.lookupTransform(odom_frame_, map_frame_,
            //                                                                             ros::Time(0));
            tf::transformMsgToTF(tfmsg_odom_map.transform, tf_odom_map_);
            ROS_INFO("Locked transform map --> odom");
            
            // tf_buffer_.canTransform(mbes_frame_, base_frame_, ros::Time(0), ros::Duration(10.));
            auto asynch_2 = std::async(std::launch::async, [this]
                                 { return tf_buffer_.lookupTransform(mbes_frame_, base_frame_,
                                                                     ros::Time(0), ros::Duration(30.)); });
            // geometry_msgs::TransformStamped tfmsg_mbes_base = tf_buffer_.lookupTransform(mbes_frame_, base_frame_,
            //                                                                              ros::Time(0));
            geometry_msgs::TransformStamped tfmsg_mbes_base = asynch_2.get();
            tf::transformMsgToTF(tfmsg_mbes_base.transform, tf_mbes_base_);
            ROS_INFO("Locked transform base --> sensor");

        }
        catch (tf::TransformException &exception)
        {
            ROS_ERROR("%s", exception.what());
            ros::Duration(1.0).sleep();
        }

        ping_sub_ = nh_->subscribe(lolo_ping_top, 10, &FixMBESLolo::pingCB, this);
        ping_pub_ = nh_->advertise<sensor_msgs::PointCloud2>(rbpf_ping_top, 100);

        ros::spin();
    }

    void pingCB(const sensor_msgs::PointCloud2ConstPtr &ping)
    {

        // Publish MBES pings
        sensor_msgs::PointCloud2 mbes_i, mbes_i_map;
        PointCloudT::Ptr mbes_i_pcl(new PointCloudT);

        // Transformation map-->mbes
        try{
            // geometry_msgs::TransformStamped tfmsg_odom_base = tf_buffer_.lookupTransform(odom_frame_, base_frame_, ros::Time(0));
            auto asynch_3 = std::async(std::launch::async, [this]
                        { return tf_buffer_.lookupTransform(odom_frame_, base_frame_,
                                                            ros::Time(0), ros::Duration(0.1)); });
            // geometry_msgs::TransformStamped tfmsg_mbes_base = tf_buffer_.lookupTransform(mbes_frame_, base_frame_,
            //                                                                              ros::Time(0));
            geometry_msgs::TransformStamped tfmsg_odom_base = asynch_3.get();
            
            tf::Transform tf_odom_base;
            tf::transformMsgToTF(tfmsg_odom_base.transform, tf_odom_base);
            tf::Transform tf_odom_mbes = tf_odom_base * tf_mbes_base_.inverse();

            // Publish pings in MBES frame
            pcl::fromROSMsg(*ping, *mbes_i_pcl);

            // For debugging
            // std::cout << "--------------" << std::endl;
            // for (auto points: mbes_i_pcl->points){
            //     std::cout << points << std::endl;
            // }
            pcl_ros::transformPointCloud(*mbes_i_pcl, *mbes_i_pcl, tf_odom_mbes.inverse());

            // Nacho: this might not be needed here?
            std::reverse(mbes_i_pcl->points.begin(), mbes_i_pcl->points.end());
            pcl::toROSMsg(*mbes_i_pcl.get(), mbes_i);
            mbes_i.header.frame_id = mbes_frame_;
            mbes_i.header.stamp = ping->header.stamp;
            ping_pub_.publish(mbes_i);
        }
        catch (tf::TransformException &exception)
        {
            ROS_ERROR("%s", exception.what());
        }
    }


    // void pingCB(const sensor_msgs::PointCloud2ConstPtr &ping)
    // {

    //     // Publish MBES pings
    //     sensor_msgs::PointCloud2 mbes_i, mbes_i_map;
    //     PointCloudT::Ptr mbes_i_pcl(new PointCloudT);
    //     std::cout << "Ping cb!" << std::endl;
        
    //     tf::Transform tf_odom_base;
    //     tf::poseMsgToTF (odom_latest_.pose.pose, tf_odom_base);
    //     tf::Transform tf_odom_mbes = tf_odom_base * tf_mbes_base_.inverse();

    //     // Publish pings in MBES frame
    //     pcl::fromROSMsg(*ping, *mbes_i_pcl);

    //     // For debugging
    //     // std::cout << "--------------" << std::endl;
    //     // for (auto points: mbes_i_pcl->points){
    //     //     std::cout << points << std::endl;
    //     // }
    //     pcl_ros::transformPointCloud(*mbes_i_pcl, *mbes_i_pcl, tf_odom_mbes.inverse());

    //     // Nacho: this might not be needed here?
    //     std::reverse(mbes_i_pcl->points.begin(), mbes_i_pcl->points.end());
    //     pcl::toROSMsg(*mbes_i_pcl.get(), mbes_i);
    //     mbes_i.header.frame_id = mbes_frame_;
    //     mbes_i.header.stamp = ping->header.stamp;
    //     ping_pub_.publish(mbes_i);
    // }

    void odom_callback(const nav_msgs::OdometryConstPtr& odom_msg)
    {
        time_ = odom_msg->header.stamp.toSec();
        odom_latest_ = *odom_msg;
    }

    std::string node_name_;
    ros::NodeHandle *nh_;

    nav_msgs::Odometry odom_latest_;
    double time_;
    double old_time_;

    tf::TransformListener tflistener_;
    tf2_ros::Buffer tf_buffer_;
    ros::Publisher ping_pub_;
    ros::Subscriber ping_sub_;

    std::string map_frame_, odom_frame_, base_frame_, mbes_frame_;
    tf::StampedTransform tf_mbes_base_;
    tf::StampedTransform tf_odom_map_;
    tf::StampedTransform tf_odom_base_;
    geometry_msgs::TransformStamped new_base_link_;
};

int main(int argc, char **argv)
{

    ros::init(argc, argv, "fix_mbes_lolo");
    ros::NodeHandle nh("~");

    FixMBESLolo *fix_mbes_lolo = new FixMBESLolo(ros::this_node::getName(), nh);
    ros::waitForShutdown();

    if (!ros::ok())
    {
        delete fix_mbes_lolo;
    }
    ROS_INFO("fix_mbes_lolo finished");

    return 0;
}