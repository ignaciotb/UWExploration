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
    RbpfParticle(int beams_num, int p_num,
                 Eigen::ArrayXXf mbes_tf_matrix, Eigen::ArrayXXf m2o_matrix,
                 vector<float> init_cov, float meas_std, vector<float> process_cov);
    ~RbpfParticle();

private:

    int p_num_;
    int index_;

    int beams_num_;
    Eigen::ArrayXXf mbes_tf_mat_;
    Eigen::ArrayXXf m2o_tf_mat_;
    float w_;
    float log_w;

    // For gp and likelihood
    Eigen::ArrayXf inputs_;
    Eigen::ArrayXf est_map_;
    Eigen::ArrayXf sigma_obs_;
    vector<float> sigma_list_;
    vector<float> mu_list_;

    // For the ancestry tree
    Eigen::ArrayXf trajectory_path_;
    bool time4regression_;
    string parent_;
    int n_from_;
    int ctr_;

    // Nacho
    vector<tuple<Eigen::ArrayXf, Eigen::ArrayXXf>> pose_history_;

    // Particle
    Eigen::VectorXd p_pose_;

    // Noise models
    std::random_device* rd_{};
    std::mt19937* seed_;
    std::vector<float> init_cov_;
    std::vector<float> meas_cov_;
    std::vector<float> process_cov_;

    void add_noise(std::vector<double>& noise);

    void motion_prediction(nav_msgs::Odometry& odom_t, int dt);

};

double angle_limit(double angle);

sensor_msgs::PointCloud2 pack_cloud(string frame, std::vector<Eigen::RowVector3f> mbes);

Eigen::ArrayXXf pcloud2ranges_full(const sensor_msgs::PointCloud2& point_cloud, int beams_num);

std::vector<int> linspace(float start, float end, float num);