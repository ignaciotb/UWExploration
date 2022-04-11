#pragma once

#include <math.h>
#include <ros/ros.h>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>

#include <std_msgs/Header.h>

#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <sensor_msgs/PointField.h>
#include <nav_msgs/Odometry.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>

#include <geometry_msgs/Point32.h>

#include <random>

using namespace std;

class RbpfParticle
{

public:
    RbpfParticle(int beams_num, int pc, int i, Eigen::Matrix4f base2mbes_mat,
                 Eigen::Matrix4f m2o_matrix, std::vector<float> init_cov, float meas_std,
                 std::vector<float> process_cov);
    ~RbpfParticle();
    
    void add_noise(std::vector<float> &noise);

    void motion_prediction(nav_msgs::Odometry &odom_t, float dt);

    void update_pose_history();

    void get_p_mbes_pose();

private:

    // Particle
    Eigen::VectorXf p_pose_;
    int beams_num_; 
    int p_num_;
    int index_; 

    Eigen::Matrix4f mbes_tf_matrix_;
    Eigen::Matrix4f m2o_matrix_;

    // Noise models
    std::random_device *rd_{};
    std::mt19937* seed_;
    std::vector<float> init_cov_;
    std::vector<float> meas_cov_;
    std::vector<float> process_cov_;

    double w_;
    std::vector<Eigen::VectorXf, Eigen::aligned_allocator<Eigen::VectorXf>> pos_history_;
    std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> rot_history_;

};

float angle_limit(float angle);

sensor_msgs::PointCloud2 pack_cloud(string frame, std::vector<Eigen::RowVector3f> mbes);

Eigen::ArrayXXf pcloud2ranges_full(const sensor_msgs::PointCloud2& point_cloud, int beams_num);

std::vector<int> linspace(float start, float end, float num);