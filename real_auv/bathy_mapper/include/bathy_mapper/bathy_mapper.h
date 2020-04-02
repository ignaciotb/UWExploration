#ifndef BATHY_MAPPER_H
#define BATHY_MAPPER_H

#include <time.h>

#include <ros/ros.h>

#include <ufomap/octree.h>
#include <bathy_mapper/ServerConfig.h>

#include <dynamic_reconfigure/server.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <tf_conversions/tf_eigen.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <eigen_conversions/eigen_msg.h>

#include "auv_2_ros/MbesSimAction.h"
#include <actionlib/server/simple_action_server.h>

#include "data_tools/std_data.h"
#include "submaps_tools/cxxopts.hpp"
#include "submaps_tools/submaps.hpp"

namespace bathy_mapper
{
class BathyMapper
{
public:
    BathyMapper(ros::NodeHandle& nh, ros::NodeHandle& nh_priv);

    ~BathyMapper();

private:
	void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg);

    void timerCallback(const ros::TimerEvent&);

    void configCallback(bathy_mapper::ServerConfig& config, uint32_t level);

    void simulateMBES(const auv_2_ros::MbesSimGoalConstPtr &mbes_goal);

    void saveMap(std::string& filename);

    void loadMap(std::string& filename);

    void init(const boost::filesystem::path map_path);


private:
	ros::NodeHandle& nh_;
	ros::NodeHandle& nh_priv_;

	ros::Subscriber cloud_sub_;

	ros::Publisher map_pub_;
	ros::Publisher map_binary_pub_;
	ros::Publisher cloud_pub_;

	ros::Timer pub_timer_;
    ros::Timer pub_timer_2_;

	ros::ServiceServer get_map_server_;
	ros::ServiceServer clear_area_server_;
	ros::ServiceServer reset_server_;

	// TF2
	tf2_ros::Buffer tf_buffer_;
	tf2_ros::TransformListener tf_listener_;

	// Dynamic reconfigure
    dynamic_reconfigure::Server<bathy_mapper::ServerConfig> cs_;
    dynamic_reconfigure::Server<bathy_mapper::ServerConfig>::CallbackType f_;

	ufomap::Octree map_;

	// Configureable variables

	std::string frame_id_;
    std::string out_map_file_, in_map_file_;

	float max_range_;

	bool insert_discrete_;
	unsigned int insert_depth_;
	unsigned int insert_n_;
	bool clear_robot_enabled_;

	float robot_height_;
	float robot_radius_;

	float pub_rate_;

	ros::Duration transform_timeout_;

	unsigned int cloud_in_queue_size_;
	unsigned int map_queue_size_;
	unsigned int map_binary_queue_size_;
	unsigned int map_cloud_queue_size_;

    ufomap::Point3 sensor_origin_;
    Eigen::Quaterniond sensor_orientation_;
    Eigen::Quaterniond q_180_;
    std::vector<ufomap::Point3 > beam_directions_;
    std::vector<ufomap::Point3> ping_;
//    ufomap::PointCloud ping_cloud_;
    sensor_msgs::PointCloud2 ping_cloud_;
    float spam_;
    float n_beams_;

    actionlib::SimpleActionServer<auv_2_ros::MbesSimAction>* as_;
    std::string action_name_;
    auv_2_ros::MbesSimFeedback feedback_;
    auv_2_ros::MbesSimResult result_;

    std::string world_frame_, map_frame_, odom_frame_, base_frame_, mbes_frame_;

    SubmapsVec maps_gt_;
    tf2_ros::StaticTransformBroadcaster static_broadcaster_;
    Eigen::Isometry3d map_tf_;

    double time_avg_;
    int iter_;
};
}  // namespace bathy_mapper

#endif  // BATHY_MAPPER_H
