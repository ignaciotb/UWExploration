#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <chrono>
#include <ros/ros.h>
#include <std_msgs/Header.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/PointField.h>
#include <actionlib/server/simple_action_server.h>
#include <auv_model/MbesSimAction.h>
#include <auv_model/SssSimAction.h>
#include <auv_model/SssSimResult.h>
#include <auv_model/Sidescan.h>
#include <Eigen/Dense>

#include "data_tools/csv_data.h"
#include "data_tools/xtf_data.h"
#include <bathy_maps/base_draper.h>
#include <bathy_maps/mesh_map.h>

#include <opencv2/opencv.hpp>

// #include "cnpy.h"
// #include <igl/embree/EmbreeIntersector.h>
#include "auv_model/sss_payload.hpp"


class AuvPayload {

public:
    
    AuvPayload(ros::NodeHandle& nh) 
    {
        // Load parameters
        std::string mesh_resources_path;
        nh.getParam("mesh_resources", mesh_resources_path);
        drap_wrap = std::make_unique<DraperWrapper>(mesh_resources_path);

        std::string sim_sss_as;
        nh.getParam("sss_sim_as", sim_sss_as);
        bool server_mode = nh.param("server_mode", false);
        as_ping = std::make_unique<actionlib::SimpleActionServer<auv_model::SssSimAction>>(
            nh, sim_sss_as, boost::bind(&AuvPayload::sssAsCallback, this, _1), server_mode
        );

        ROS_INFO("Initialized AUV payload Node.");
        ros::spin();
    }

private:

    
    void sssAsCallback(const auv_model::SssSimGoalConstPtr& goal) 
    {
        Eigen::Vector3d position(goal->sss_pose.transform.translation.x,
                                  goal->sss_pose.transform.translation.y,
                                  goal->sss_pose.transform.translation.z);

        Eigen::Quaterniond quaternion(
            goal->sss_pose.transform.rotation.w,
            goal->sss_pose.transform.rotation.x,
            goal->sss_pose.transform.rotation.y,
            goal->sss_pose.transform.rotation.z);

        Eigen::Vector3d euler_angles = quaternion.toRotationMatrix().eulerAngles(0, 1, 2);

        xtf_data::xtf_sss_ping xtf_ping;
        xtf_ping.port.time_duration = drap_wrap->max_r * 2 / drap_wrap->svp;
        xtf_ping.stbd.time_duration = drap_wrap->max_r * 2 / drap_wrap->svp;
        xtf_ping.pos_ = position;
        xtf_ping.roll_ = euler_angles[0];
        xtf_ping.pitch_ = euler_angles[1];
        xtf_ping.heading_ = euler_angles[2];

        auto start_time = std::chrono::steady_clock::now();
        size_t nbr_bins = goal->beams_num.data;
        // Eigen::VectorXd left, right;
        ping_draping_result left, right;
        std::tie(left, right) = drap_wrap->draper->project_ping(xtf_ping, nbr_bins);

        auto end_time = std::chrono::steady_clock::now();
        avg_time.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count());
        if (avg_time.size() > 10.) avg_time.erase(avg_time.begin());

        auv_model::Sidescan sss_msg;
        sss_msg.port_channel.resize(left.time_bin_model_intensities.size());
        std::transform(left.time_bin_model_intensities.data(), left.time_bin_model_intensities.data() + left.time_bin_model_intensities.size(), 
                        sss_msg.port_channel.begin(), [](double value)
                                    { return static_cast<uint8_t>(std::round(value * 255.0)); });

        sss_msg.starboard_channel.resize(right.time_bin_model_intensities.size());
        std::transform(right.time_bin_model_intensities.data(), right.time_bin_model_intensities.data() + right.time_bin_model_intensities.size(),
                       sss_msg.starboard_channel.begin(), [](double value)
                       { return static_cast<uint8_t>(std::round(value * 255.0)); });

        std::cout << "Ping projection time (ms) " << std::accumulate(avg_time.begin(), avg_time.end(), 0) / 10. << std::endl;

        auv_model::SssSimResult result;
        result.sim_sss = sss_msg;
        as_ping->setSucceeded(result);
    }

    double mbes_angle;
    std::unique_ptr<DraperWrapper> drap_wrap;
    std::vector<double> avg_time;
    std::unique_ptr<actionlib::SimpleActionServer<auv_model::SssSimAction>> as_ping;
};

int main(int argc, char** argv) 
{
    ros::init(argc, argv, "auv_payload");
    ros::NodeHandle nh("~");
    try {
        AuvPayload model(nh);
    } catch (const std::exception& e) {
        ROS_ERROR("Error initializing AUV payload model: %s", e.what());
    }
    return 0;
}
