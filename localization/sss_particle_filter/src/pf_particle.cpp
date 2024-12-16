#include "sss_particle_filter/pf_particle.h"

pfParticle::pfParticle(int beams_num, int p_num, int index, Eigen::Matrix4f mbes_tf_matrix,
                           Eigen::Matrix4f m2o_matrix, Eigen::Matrix<float, 6, 1> init_pose, std::vector<float> init_cov, float meas_std,
                           std::vector<float> process_cov)
{
    p_num_ = p_num;
    index_ = index;
    beams_num_ = beams_num;
    // p_pose_ = Eigen::VectorXf::Zero(6,1);
    p_pose_ = init_pose;
    mbes_tf_matrix_ = mbes_tf_matrix;
    m2o_matrix_ = m2o_matrix;

    // Noise models init
    init_cov_ = init_cov;
    process_cov_ = process_cov;
    noise_vec_ = Eigen::VectorXf(6);
    // meas_cov_ = std::vector<double>(beams_num_, std::pow(meas_std, 2));
    mbes_sigma_ = double(meas_std);
    // Eigen::VectorXd meas_cov_eig_diag = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(meas_cov_.data(), meas_cov_.size());
    // meas_cov_mat_ = meas_cov_eig_diag.asDiagonal();

    this->add_noise(init_cov_);

    // Init particle history
    pos_history_.emplace_back(std::shared_ptr<pos_track>(new pos_track()));
    rot_history_.emplace_back(std::shared_ptr<rot_track> (new rot_track()));

    // The lock
    pc_mutex_ = std::shared_ptr<std::mutex>(new std::mutex());
}

pfParticle::~pfParticle()
{

}

void pfParticle::add_noise(std::vector<float> &noise){

    // Generate noise
    // Eigen::VectorXf noisy_pose(6, 1);
    std::random_device rd{};
    std::mt19937 seed{rd()};
    for (int i = 0; i < 6; i++)
    {
        std::normal_distribution<float> sampler{0, std::sqrt(noise.at(i))};
        p_pose_(i) += sampler(seed);
    }
}

void pfParticle::motion_prediction(Eigen::Vector3f &vel_rot, Eigen::Vector3f &vel_p, 
                                     float depth, float dt, std::mt19937& rng)
{
    for (int j = 0; j < 6; j++)
    {
        std::normal_distribution<float> sampler(0, std::sqrt(process_cov_.at(j)));
        noise_vec_(j) = sampler(rng);
    }

    Eigen::Vector3f rot_t = p_pose_.tail(3) + vel_rot * dt + noise_vec_.tail(3);
    // Wrap up angles
    for (int i = 0; i < 3; i++)
    {
        rot_t(i) = angle_limit(rot_t(i));
    }
    p_pose_.tail(3) = rot_t;
    p_pose_(3) = 0.;
    p_pose_(4) = 0.;

    Eigen::AngleAxisf rollAngle(rot_t(0), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle(rot_t(1), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle(rot_t(2), Eigen::Vector3f::UnitZ());
    Eigen::Quaternion<float> q = rollAngle * pitchAngle * yawAngle;

    Eigen::Vector3f step_t = q.matrix() * (vel_p * dt) + noise_vec_.head(3);
    p_pose_.head(3) += step_t;
    // NACHO: read depth directly from DR
    p_pose_(2) = depth;
}

void pfParticle::update_pose_history()
{
    // Rotation matrix
    std::lock_guard<std::mutex> lock(*pc_mutex_);
    Eigen::AngleAxisf rollAngle_p(p_pose_(3), Eigen::Vector3f::UnitX());
    Eigen::AngleAxisf pitchAngle_p(p_pose_(4), Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf yawAngle_p(p_pose_(5), Eigen::Vector3f::UnitZ());
    Eigen::Quaternion<float> q_p = rollAngle_p * pitchAngle_p * yawAngle_p;

    // Particle pose in homogenous coordinates
    Eigen::Matrix4f t_p = Eigen::Matrix4f::Identity();
    t_p.topLeftCorner(3, 3) = q_p.matrix();
    t_p.block(0, 3, 3, 1) = p_pose_.head(3);
    Eigen::Matrix4f p_pose_map = m2o_matrix_ * t_p * mbes_tf_matrix_;

    pos_history_.back()->push_back(p_pose_map.block(0, 3, 3, 1));
    rot_history_.back()->push_back(p_pose_map.topLeftCorner(3, 3));
}

void pfParticle::compute_weight(Eigen::VectorXd exp_mbes, Eigen::VectorXd real_mbes)
{
    // Compare only absolute z values of real and expected measurements
    // vector<float> exp_mbes_z = list2ranges(exp_mbes);
    if (exp_mbes.size() > 0)
    {
        w_ = weight_mv(real_mbes, exp_mbes);
    }
    else
    {
        w_ = 1e-50;
        ROS_WARN("Range of exp meas equals zero");
    }
}

double pfParticle::weight_mv(Eigen::VectorXd& mbes_meas_ranges, Eigen::VectorXd& mbes_sim_ranges)
{
    float w_i;
    if(mbes_meas_ranges.size() == mbes_sim_ranges.size())
    {
        w_i = log_pdf_uncorrelated(mbes_meas_ranges, mbes_sim_ranges, gp_covs_, mbes_sigma_);

        // std::vector<double> meas_cov_ = std::vector<double>(beams_num_, std::pow(mbes_sigma_, 2));
        // Eigen::VectorXd meas_cov_eig_diag = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(meas_cov_.data(), meas_cov_.size());
        // Eigen::MatrixXd meas_cov_mat_ = meas_cov_eig_diag.asDiagonal();
        // w_i = mvn_pdf(mbes_meas_ranges, mbes_sim_ranges, meas_cov_mat_);
        
        if (!std::isfinite(w_i)){
            w_i = 1e-50;
            ROS_WARN("Nan weights!");
        }
    }

    else
    {
        ROS_WARN("Missing pings!");
        w_i = 1e-50;
    }

    return w_i + 1.e-200;
}

// TODO: if we don't use covs from the GP maps, define sigma.inverse() in the class constructor
float mvn_pdf(const Eigen::VectorXd& x, Eigen::VectorXd& mean, Eigen::MatrixXd& sigma)
{
    float quadform  = (x - mean).transpose() * sigma.inverse() * (x - mean);
    float norm = 1.0 / std::sqrt((2*M_PI*sigma).determinant());

    return norm * exp(-0.5 * quadform);
}

double log_pdf_uncorrelated(const Eigen::VectorXd &x, Eigen::VectorXd &mean,
                            Eigen::VectorXd &gp_var, double &mbes_sigma)
{
    double n = double(x.cols());
    
    // Set GP variances to zero (for testing)
    // gp_var.setZero();

    // Eigen::VectorXd sigmas_2 = gp_var.array().square() + std::pow(mbes_sigma, 2);
    // Eigen::VectorXd diff = (x - mean).array().square();
    // diff.array() *= ((2. * sigmas_2.array()).inverse()).transpose(); // Nasty inverse
    // double logl = -(1./2.)*(sigmas_2.array().log()).sum() - (n / 2.0) 
    //                 * std::log(2 * M_PI) - diff.array().sum();

    // std::cout << (x - mean).array().sum() << std::endl;
    Eigen::VectorXd var_diag = gp_var.array() + std::pow(mbes_sigma, 2);
    Eigen::MatrixXd var_inv = var_diag.cwiseInverse().asDiagonal();
    Eigen::MatrixXd var_mat = var_diag.asDiagonal();
    Eigen::VectorXd diff = (x - mean).array().transpose() * var_inv.array() * 
                            (x - mean).array();
    double logl = -(n / 2.) * std::log(var_mat.determinant()) 
                  -(1 / 2.0) * diff.array().sum();

    return exp(logl);
    // return 1./logl;
}

vector<float> list2ranges(vector<Eigen::Array3f> points)
{
    vector<float> ranges;
    for(int i = 0; i < points.size(); i++) { ranges[i] = points[i](2); }
    return ranges;
}


float angle_limit(float angle) // keep angle within [0;2*pi]
{
    while (angle >= M_PI*2)
        angle -= M_PI*2;
    while (angle < 0)
        angle += M_PI*2;
    return angle;
}

sensor_msgs::PointCloud2 pack_cloud(string frame, std::vector<Eigen::RowVector3f> mbes)
{
    sensor_msgs::PointCloud2 mbes_pcloud;
    PCloud::Ptr pcl_pcloud(new PCloud);
    pcl_pcloud->header.frame_id = frame;
    // pcl_pcloud->height = pcl_pcloud->width = 1;
    for(int i = 0; i < mbes.size(); i++)
    {
        pcl_pcloud->points.push_back(pcl::PointXYZ(mbes[i][0], mbes[i][1], mbes[i][2]));
    }

    pcl::toROSMsg(*pcl_pcloud.get(), mbes_pcloud);

    return mbes_pcloud;
}

// Eigen::ArrayXXf pcloud2ranges_full(const sensor_msgs::PointCloud2& point_cloud, int beams_num)
// {
//     // Nacho: is this needed?
//     sensor_msgs::PointCloud out_cloud;
//     sensor_msgs::convertPointCloud2ToPointCloud(point_cloud, out_cloud);

//     // Selecting only self.beams_num of beams in the ping
//     std::vector<int> idx = linspace(0, point_cloud.width - 1, beams_num);
//     Eigen::MatrixXf beams(beams_num, 3);
//     int beam_cnt = 0;
//     for (int i = 0; i < out_cloud.points.size(); ++i)
//     {
//         if (std::find(idx.begin(), idx.end(), i) != idx.end() ){
//             beams.row(beam_cnt) = Eigen::Vector3f(out_cloud.points[i].x,
//                                                   out_cloud.points[i].y,
//                                                   out_cloud.points[i].z);
//             beam_cnt++;
//         }
//     }
//     return beams;
// }

Eigen::MatrixXf Pointcloud2msgToEigen(const sensor_msgs::PointCloud2 &cloud, int beams_num)
{
    sensor_msgs::PointCloud2::Ptr cloud_ptr(new sensor_msgs::PointCloud2(cloud));
    sensor_msgs::PointCloud2Iterator<float> iter_x(*cloud_ptr, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*cloud_ptr, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*cloud_ptr, "z");

    // Selecting only self.beams_num of beams in the ping
    std::vector<int> idx = linspace(0, cloud.width - 1, beams_num);
    Eigen::MatrixXf mat(beams_num, 3);

    int beam_cnt = 0;
    for (size_t i = 0; i < cloud.width; ++i, ++iter_x, ++iter_y, ++iter_z)
    {
        if (std::find(idx.begin(), idx.end(), i) != idx.end())
        {
            mat.row(beam_cnt)[0] = *iter_x;
            mat.row(beam_cnt)[1] = *iter_y;
            mat.row(beam_cnt)[2] = *iter_z;
            beam_cnt++;
        }
    }
    return mat;
}

void eigenToPointcloud2msg(sensor_msgs::PointCloud2 &cloud, Eigen::MatrixXf &mat)
{

    sensor_msgs::PointCloud2Modifier cloud_out_modifier(cloud);
    cloud_out_modifier.setPointCloud2FieldsByString(1, "xyz");
    cloud_out_modifier.resize(mat.rows());

    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");

    for (size_t i = 0; i < mat.rows(); ++i, ++iter_x, ++iter_y, ++iter_z)
    {
        *iter_x = mat.row(i)[0];
        *iter_y = mat.row(i)[1];
        *iter_z = mat.row(i)[2];
    }
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
