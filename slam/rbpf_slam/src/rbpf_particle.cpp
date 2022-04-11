#include "rbpf_particle.h"

RbpfParticle::RbpfParticle(int beams_num, int p_num,
                           Eigen::ArrayXXf mbes_tf_matrix, Eigen::ArrayXXf m2o_matrix,
                           vector<float> init_cov, float meas_std, vector<float> process_cov) : beams_num_(beams_num), p_num_(p_num), mbes_tf_mat_(mbes_tf_matrix), m2o_tf_mat_(m2o_matrix),
                                                                                                init_cov_(init_cov), process_cov_(process_cov)
{
    p_pose_ = Eigen::VectorXd::Zero(6, 1);
}

RbpfParticle::~RbpfParticle()
{
}

void RbpfParticle::add_noise(std::vector<double> &noise){

    // Generate noise
    Eigen::VectorXd noisy_pose(6, 1);
    for (int i = 0; i < 6; i++)
    {
        std::normal_distribution<double> x_sampler{0, std::sqrt(noise.at(i))};
        p_pose_(i) += x_sampler(*seed_);
    }
}

void RbpfParticle::motion_prediction(nav_msgs::Odometry &odom_t, int dt){

    // Generate noise
    Eigen::VectorXd noise_vec(6, 1);
    for (int i = 0; i < 6; i++)
    {
        std::normal_distribution<double> x_sampler{0, std::sqrt(process_cov_.at(i))};
        noise_vec(i) = x_sampler(*seed_);
    }

    // Angular 
    Eigen::Vector3d vel_rot = Eigen::Vector3d(odom_t.twist.twist.angular.x,
                                              odom_t.twist.twist.angular.y,
                                              odom_t.twist.twist.angular.z);

    Eigen::Vector3d rot_t = p_pose_.tail(3) + vel_rot * dt + noise_vec.tail(3);
    // Wrap up angles
    for (int i =0; i < 3; i++){
        rot_t(i) = angle_limit(rot_t(i));
    }
    p_pose_.tail(3) = rot_t;


    Eigen::AngleAxisd rollAngle(rot_t(0), Eigen::Vector3d::UnitZ());
    Eigen::AngleAxisd pitchAngle(rot_t(1), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd yawAngle(rot_t(2), Eigen::Vector3d::UnitY());
    Eigen::Quaternion<double> q = rollAngle * yawAngle * pitchAngle;
    Eigen::Matrix3d rotMat = q.matrix();

    // Linear
    Eigen::Vector3d vel_p = Eigen::Vector3d(odom_t.twist.twist.linear.x,
                                            odom_t.twist.twist.linear.y,
                                            odom_t.twist.twist.linear.z);
    
    Eigen::Vector3d step_t = rotMat * vel_p * dt + noise_vec.head(3);
    p_pose_.head(3) += step_t;
}

double angle_limit(double angle) // keep angle within [0;2*pi]
{
    while (angle >= M_PI*2)
        angle -= M_PI*2;
    while (angle < 0)
        angle += M_PI*2;
    return angle;
}

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

Eigen::ArrayXXf pcloud2ranges_full(const sensor_msgs::PointCloud2& point_cloud, int beams_num)
{
    int pc_size = point_cloud.row_step;

    // Nacho: is this needed?
    sensor_msgs::PointCloud out_cloud;
    sensor_msgs::convertPointCloud2ToPointCloud(point_cloud, out_cloud);

    // Selecting only self.beams_num of beams in the ping
    std::vector<int> idx;
    idx = linspace(0, point_cloud.row_step - 1, beams_num);
    Eigen::MatrixXf beams(beams_num, 3);

    for (int i = 0; i < out_cloud.points.size(); ++i)
    {
        if (std::find(idx.begin(), idx.end(), i) != idx.end() ){
            beams.row(i) = Eigen::Vector3f(out_cloud.points[i].x, 
                                            out_cloud.points[i].y,
                                            out_cloud.points[i].z);
        }
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