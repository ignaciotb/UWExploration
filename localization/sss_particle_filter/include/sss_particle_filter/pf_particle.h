#pragma once

#include <math.h>
#include <tf/tf.h>
#include <ros/ros.h>
#include <cmath>
#include <chrono>
#include <mutex>
#include <random>

#include <Eigen/Dense>
#include <Eigen/Core>
#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>

#include <std_msgs/Header.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <sensor_msgs/PointField.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Point32.h>

#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

// #include "data_tools/csv_data.h"
// #include "data_tools/xtf_data.h"
// #include <bathy_maps/base_draper.h>
// #include <bathy_maps/mesh_map.h>
#include "sss_particle_filter/sss_payload.hpp"
#include <auv_model/Sidescan.h>

using namespace std;
using namespace cv;

typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> pos_track;
typedef std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f>> rot_track;
typedef std::shared_ptr<pos_track> pos_track_ptr;
typedef std::shared_ptr<rot_track> rot_track_ptr;

typedef pcl::PointCloud<pcl::PointXYZ> PCloud;

class pfParticle
{

public:
    pfParticle(int beams_num, int pc, int i, Eigen::Matrix4f base2mbes_mat,
               Eigen::Matrix4f m2o_matrix, Eigen::Matrix<float, 6, 1> init_pose, std::vector<float> init_cov, float meas_std,
               std::vector<float> process_cov, std::string mesh_resources_path, string results_path);
    ~pfParticle();

    void add_noise(std::vector<float> &noise);

    void compute_weight(Eigen::VectorXd exp_mbes, Eigen::VectorXd real_mbes);

    double weight_mv(Eigen::VectorXd &mbes_meas_ranges, Eigen::VectorXd &mbes_sim_ranges);

    void motion_prediction(Eigen::Vector3f &vel_rot, Eigen::Vector3f &vel_p, 
                                     float depth, float dt, std::mt19937& rng);
    void update_pose_history();

    void compute_weight_sss(const cv::Mat real_sss_patch);

    void sss_prediction(Eigen::Matrix4f& p_pose_map);

    double getMSSIM(const Mat &i1, const Mat &i2);

    Eigen::VectorXf p_pose_;
    std::vector<pos_track_ptr> pos_history_;
    std::vector<rot_track_ptr> rot_history_;
    double w_;
    int index_;
    int submap_cnt_;

    // Noise models
    std::vector<float> init_cov_;
    Eigen::VectorXd gp_covs_;
    std::vector<float> process_cov_;
    double mbes_sigma_;
    std::shared_ptr<std::mutex> pc_mutex_;
    Eigen::VectorXf noise_vec_;
    cv::Mat sss_patch_;
    std::shared_ptr<DraperWrapper> drap_wrap_;
    auv_model::Sidescan sss_msg_;
    string results_path_;

private:
    // vector<tuple<Eigen::ArrayXf, Eigen::ArrayXXf>> pose_history_;

    // Particle
    int beams_num_;
    int p_num_;

    Eigen::Matrix4f mbes_tf_matrix_;
    Eigen::Matrix4f m2o_matrix_;
};

float angle_limit(float angle);

sensor_msgs::PointCloud2 pack_cloud(string frame, std::vector<Eigen::RowVector3f> mbes);

// Eigen::ArrayXXf pcloud2ranges_full(const sensor_msgs::PointCloud2& point_cloud, int beams_num);

Eigen::MatrixXf Pointcloud2msgToEigen(const sensor_msgs::PointCloud2 &cloud, int beams_num);

void eigenToPointcloud2msg(sensor_msgs::PointCloud2 &cloud, Eigen::MatrixXf &mat);

vector<float> list2ranges(vector<Eigen::Array3f> points);

std::vector<int> linspace(float start, float end, float num);

float mvn_pdf(const Eigen::VectorXd& x, Eigen::VectorXd& mean, Eigen::MatrixXd& sigma);

double log_pdf_uncorrelated(const Eigen::VectorXd &x, Eigen::VectorXd &mean,
                            Eigen::VectorXd &gp_sigmas, double &mbes_sigma);

// struct normal_random_variable_
// {

//     normal_random_variable_(Eigen::VectorXd const &mean, Eigen::MatrixXd const &covar)
//         : mean(mean)
//     {
//         Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
//         transform = eigenSolver.eigenvectors() * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();
//     }

//     Eigen::VectorXd mean;
//     Eigen::MatrixXd transform;

//     Eigen::VectorXd operator()() const
//     {
//         static std::mt19937 gen{std::random_device{}()};
//         static std::normal_distribution<> dist;

//         return mean + transform * Eigen::VectorXd{mean.size()}.unaryExpr([&](auto x) { return dist(gen); });
//     }
// };
