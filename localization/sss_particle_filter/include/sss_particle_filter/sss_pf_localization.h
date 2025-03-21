// HEADER DEFINING THE RBPF_SLAM CLASS
#pragma once

#include "sss_particle_filter/pf_particle.h"

// Standard dependencies
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <set>
#include <thread>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf/transform_listener.h>

#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Float32MultiArray.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Int32MultiArray.h>
#include <std_msgs/String.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>

#include <auv_model/Sidescan.h>
#include <auv_model/SssSimAction.h>

#include <std_srvs/Empty.h>

#include <actionlib/server/simple_action_server.h>
#include <actionlib/client/simple_action_client.h>
#include <slam_msgs/MinibatchTrainingAction.h>
#include <slam_msgs/MinibatchTrainingGoal.h>
#include <slam_msgs/MinibatchTrainingResult.h>
#include <slam_msgs/SamplePosteriorAction.h>
#include <slam_msgs/ManipulatePosteriorAction.h>
#include <slam_msgs/PlotPosteriorAction.h>
#include <slam_msgs/Resample.h>

#include "tf/transform_datatypes.h"
#include "tf_conversions/tf_eigen.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>
#include <future>


using namespace std;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

// typedef actionlib::SimpleActionClient<slam_msgs::SamplePosteriorAction> Client;

class pfLocalization
{

public:

    pfLocalization();
    pfLocalization(ros::NodeHandle &nh, ros::NodeHandle &nh_mb);
    ~pfLocalization();

    Eigen::Matrix<float, 6, 1> init_p_pose_;

    ros::NodeHandle *nh_;
    ros::NodeHandle *nh_mb_;
    std::string node_name_;
    ros::Timer timer_rbpf_, timer_rviz_;

    // Multithreading
    std::vector<std::thread> pred_threads_vec_;
    std::vector<std::thread> meas_threads_vec_;
    std::vector<std::thread> upd_threads_vec_;
    std::vector<std::thread> weights_threads_vec_;

    int pc_;
    int beams_num_;
    int sss_bin_num_;
    int beams_real_;
    float mbes_angle_;
    string map_frame_;
    string base_frame_;
    string sss_frame_;
    string odom_frame_;
    std::vector<pfParticle> particles_;
    std::vector<pfParticle> dr_particle_;
    std::mt19937 rng_;
    std::mt19937 g_;
    int ping_cnt_;
    int submap_size_;

    tf::TransformListener tfListener_;
    tf2_ros::Buffer tf_buffer_;
    
    bool lc_detected_;
    bool start_training_;
    double time_avg_;
    int count_mb_cbs_;
    int count_mbes_cbs_;

    // Covariances
    float meas_std_;
    std::vector<float> init_cov_;
    std::vector<float> res_noise_cov_;
    std::vector<float> motion_cov_;
    Eigen::Matrix3f cov_;

    // Global variables
    // std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> sss_history_;
    cv::Mat sss_full_image_;
    vector<int> n_eff_mask_;
    vector<float> pw_;
    std::vector<int> pings_idx_;
    std::vector<int> ancestry_sizes_;
    std::vector<int> beams_idx_;
    std::vector<int> svgp_lc_ready_;
    Eigen::VectorXf latest_mbes_z_;

    float n_eff_filt_;
    int N_eff_;
    int count_pings_;
    bool survey_finished_;
    double time_;
    double old_time_;
    float rbpf_period_, rviz_period_;

    sensor_msgs::PointCloud2 prev_mbes_;
    sensor_msgs::PointCloud2 latest_mbes_;
    geometry_msgs::PoseWithCovarianceStamped avg_pose_;
    Eigen::MatrixXf ping_mat_;
    Eigen::MatrixXf mb_mat_;
    bool first_update_;

    // Publishers
    ros::Publisher ip_pub_;
    std::vector<ros::Publisher> p_sss_pubs_;
    std::vector<ros::ServiceClient> p_resampling_srvs_;
    ros::Publisher pf_pub_;
    ros::Publisher avg_pub_;
    ros::Publisher pf_mbes_pub_;
    ros::Publisher stats_;
    ros::Publisher dr_estimate_pub_;
    ros::Publisher gp_saved_pub_;
    ros::Publisher vis_pub_;

    string pose_array_top_;
    string avg_pose_top_;
    string pf_mbes_top_;
    string stats_top_;
    string mbes_pc_top_;

    std::vector<double> avg_time;
    
    // // Minibatch AS
    // actionlib::SimpleActionServer<slam_msgs::MinibatchTrainingAction>* as_mb_;
    // string mb_gp_name_;

    // // Action clients for manipulating the the GPs posteriors
    // std::vector<actionlib::SimpleActionClient<slam_msgs::ManipulatePosteriorAction>*> p_manipulate_acs_;
    // string manipulate_gp_server_;
    std::vector<int> updated_w_ids_;
    // std::vector<int> updated_saved_ids_;

    actionlib::SimpleActionClient<auv_model::SssSimAction> *ac_sss_;

    // Server
    ros::ServiceServer srv_server_;
    string synch_top_;

    // Subscribers
    ros::Subscriber mbes_sub_;
    ros::Subscriber sss_sub_;
    ros::Subscriber odom_sub_;
    ros::Subscriber finished_sub_;
    ros::Subscriber save_sub_;
    ros::Subscriber lc_manual_sub_;
    ros::Subscriber enable_lc_sub_;
    ros::Subscriber path_sub_;

    // typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::PointCloud2, nav_msgs::Odometry> MySyncPolicy;
    // message_filters::Synchronizer<MySyncPolicy> *synch_;

    string mbes_pings_top_;
    string sss_pings_top_;
    string odom_top_;
    string finished_top_;
    string save_top_;
    string lc_manual_topic_;
    string path_topic_;
    string pose_dr_top_;
    string sss_sim_as_;
    string particles_paths_file_;
    string results_path_;
    
    // End of mission timer
    bool mission_finished_;
    float time_wo_motion_;
    nav_msgs::Odometry odom_latest_;
    nav_msgs::Odometry odom_end_;

    // Transforms
    Eigen::Matrix4f base2mbes_mat_;
    Eigen::Matrix4f m2o_mat_;

    // Callbacks
    bool empty_srv(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res);
    void enable_lc(const std_msgs::Int32::ConstPtr& enable_lc);
    // void mb_cb(const slam_msgs::MinibatchTrainingGoalConstPtr &goal);
    void manual_lc(const std_msgs::Bool::ConstPtr& lc_msg);
    // void path_cb(const nav_msgs::PathConstPtr& wp_path);
    void synch_cb(const std_msgs::Bool::ConstPtr& finished_msg);
    void save_cb(const std_msgs::Bool::ConstPtr& save_msg);
    // void mbes_real_cb(const sensor_msgs::PointCloud2ConstPtr &msg);
    void sss_cb(const auv_model::SidescanConstPtr &msg);
    void odom_callback(const nav_msgs::Odometry::ConstPtr& odom_msg);
    void expected_measurements(nav_msgs::Odometry &odom);
    // void meas_predict_particles();
    void compute_weights(const cv::Mat real_sss_patch);

    // void sampleCB(const actionlib::SimpleClientGoalState &state, const slam_msgs::ManipulatePosteriorResultConstPtr &result);
    // void measCB(const sensor_msgs::PointCloud2ConstPtr &mbes_ping,
    //             const nav_msgs::OdometryConstPtr &odom_msg);

    // Other functions
    void setup_svgps();
    // void pf_update(const ros::TimerEvent&);
    void update_rviz(const ros::TimerEvent &);
    // void update_particles_weights(sensor_msgs::PointCloud2 &mbes_ping, nav_msgs::Odometry& odom);
    void save_paths(const bool plot);
    void predict(nav_msgs::Odometry odom_t, float dt);
    void update_particles_history();
    void publish_stats(nav_msgs::Odometry gt_odom);
    float moving_average(vector<int> a, int n);
    void resample(vector<double> weights);
    void reassign_poses(vector<int> lost, vector<int> dupes);
    vector<int> systematic_resampling(vector<double> weights);
    vector<int> arange(int start, int stop, int step);
    void average_pose(geometry_msgs::PoseArray pose_list);
    void pub_markers(const geometry_msgs::PoseArray &array_msg);
};
