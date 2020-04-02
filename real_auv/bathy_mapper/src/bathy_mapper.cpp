#include <bathy_mapper/bathy_mapper.h>

#include <ufomap_msgs/Ufomap.h>
#include <ufomap_msgs/conversions.h>
#include <ufomap_ros/conversions.h>

#include <future>

namespace bathy_mapper
{
BathyMapper::BathyMapper(ros::NodeHandle& nh, ros::NodeHandle& nh_priv)
    : nh_(nh)
    , nh_priv_(nh_priv)
    , map_pub_(nh_priv.advertise<ufomap_msgs::Ufomap>(
                "map", nh_priv.param("map_queue_size", 10), nh_priv.param("map_latch", false)))
    , map_binary_pub_(nh_priv.advertise<ufomap_msgs::Ufomap>(
                "map_binary", nh_priv.param("map_binary_queue_size", 10),
                nh_priv.param("map_binary_latch", false)))
    , cloud_pub_(nh_priv.advertise<sensor_msgs::PointCloud2>(
                "map_cloud", nh_priv.param("map_cloud_queue_size", 10),
                nh_priv.param("map_cloud_latch", false)))
    , tf_listener_(tf_buffer_)
    , cs_(nh_priv)
    , map_(nh_priv.param("resolution", 0.1), nh_priv.param("depth_levels", 16),
                 !nh_priv.param("multithreaded", false))
{
    // Set up dynamic reconfigure server
    f_ = boost::bind(&BathyMapper::configCallback, this, _1, _2);
    cs_.setCallback(f_);

    cloud_sub_ = nh.subscribe("cloud_in", nh_priv.param("cloud_in_queue_size", 1000),
                                                        &BathyMapper::cloudCallback, this);

    std::string map_str, gt_pings_top, debug_pings_top, map_top, gt_odom_top, mbes_sim_as;
    nh_priv_.param<std::string>("output_map", out_map_file_, "output_map");
    nh_priv_.param<std::string>("input_map", in_map_file_, "input_map");
    nh_priv_.param<std::string>("world_frame", world_frame_, "world");
    nh_priv_.param<std::string>("map_frame", map_frame_, "map");
    nh_priv_.param<std::string>("mbes_link", mbes_frame_, "mbes_link");
    nh_priv_.param<std::string>("map_pcl", map_top, "/map");
    nh_priv_.param<float>("mbes_open_angle", spam_, 1.5708);
    nh_priv_.param<float>("num_beams_sim", n_beams_, 100);
    nh_priv_.param<std::string>("map_cereal", map_str, "map.cereal");
    nh_priv_.param<std::string>("mbes_sim_as", mbes_sim_as, "mbes_sim_action");

    boost::filesystem::path map_path(map_str);
    this->init(map_str);

    if (0 < pub_rate_)
    {
        pub_timer_ =
                nh_priv.createTimer(ros::Rate(pub_rate_), &BathyMapper::timerCallback, this);

    }

    // Load existing map
    if(!in_map_file_.empty()){
        this->loadMap(in_map_file_);
    }

    // Create and start Sim MBES as
    as_ = new actionlib::SimpleActionServer<auv_2_ros::MbesSimAction>(nh_priv_, mbes_sim_as,
                                                                      boost::bind(&BathyMapper::simulateMBES,
                                                                                  this, _1), false);
    as_->start();
}

BathyMapper::~BathyMapper(){
    if(!out_map_file_.empty()){
        this->saveMap(out_map_file_);
    }
}

void BathyMapper::init(const boost::filesystem::path map_path){

    q_180_ = Eigen::AngleAxisf(3.1415, Eigen::Vector3f::UnitX())
             * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY())
             * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitZ());

    iter_ = 0;
    time_avg_ = 0;
    ROS_INFO("Initialized Bathy mapper");
}

void BathyMapper::simulateMBES(const auv_2_ros::MbesSimGoalConstPtr &mbes_goal){

    clock_t tStart = clock();
    iter_++;

    /// Construct the MBES beams
    // set the sensor origin and sensor orientation
    Eigen::Isometry3d sensor_tf;
    tf::transformMsgToEigen(mbes_goal->mbes_pose.transform, sensor_tf);
    Eigen::Isometry3f tf = sensor_tf.cast<float>();
    sensor_origin_ = ufomap::Point3(tf.translation().x(), tf.translation().y(), tf.translation().z());
    sensor_orientation_ = Eigen::Quaternionf(tf.linear()) /** q_180_*/;

    beam_directions_.reserve(n_beams_);
    float roll_step = spam_/n_beams_;
    Eigen::Quaternionf q;
    Eigen::Matrix3f rot_beam;
    for(int i = -n_beams_/2; i<=n_beams_/2; i++){
        q = Eigen::AngleAxisf(roll_step*i, Eigen::Vector3f::UnitX())
            * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitY())
            * Eigen::AngleAxisf(0.0, Eigen::Vector3f::UnitZ());
        rot_beam = (Eigen::Quaternionf(sensor_orientation_ * q)).toRotationMatrix();
        beam_directions_.emplace_back(rot_beam.col(2).x(),rot_beam.col(2).y(),rot_beam.col(2).z()).normalize();
    }

    /// Ray trace each beam against the ufomap
    ufomap::Point3* end = new ufomap::Point3();
    ufomap::PointCloud cloud;
    cloud.reserve(beam_directions_.size() / 4);
    for(ufomap::Point3& beam_i: beam_directions_){
        // TODO: fill up beams that don't hit occupied space with sth
        if(map_.castRay(sensor_origin_, beam_i, *end, true, -1, 0)){
            cloud.push_back(*end);
        }
    }
    delete end;
    beam_directions_.clear();

    /// Action server
    if(cloud.size() == 0){
        ROS_WARN("No multibeam hits! You're out of the GT map");
        as_->setAborted(result_);
    }
    else{
//        ufomap_math::Pose6 transform = ufomap::toUfomap(mbes_goal->mbes_pose.transform);
//        cloud.transform(transform);
        sensor_msgs::PointCloud2::Ptr sim_ping(new sensor_msgs::PointCloud2);
        ufomap::fromUfomap(cloud, sim_ping);
        sim_ping->header.frame_id = mbes_goal->mbes_pose.header.frame_id;
        sim_ping->header.stamp = mbes_goal->mbes_pose.header.stamp;
        result_.sim_mbes = *sim_ping;
        as_->setSucceeded(result_);

        time_avg_ += (double)(clock() - tStart)/CLOCKS_PER_SEC;
//        printf("Sim MBES time taken: %.4fs\n", time_avg_/iter_);
    }
}

// Private functions
void BathyMapper::cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    try
    {
        ufomap::PointCloud cloud;

        // auto a1 and a2 are same as:
        // geometry_msgs::TransformStamped tmp_transform =
        // tf_buffer_.lookupTransform(frame_id_, msg->header.frame_id, msg->header.stamp,
        // transform_timeout_);
        // ufomap::toUfomap(msg, cloud);

        auto a1 = std::async(std::launch::async, [this, &msg] {
            return tf_buffer_.lookupTransform(frame_id_, msg->header.frame_id,
                                                                                msg->header.stamp, transform_timeout_);
        });
        auto a2 =
                std::async(std::launch::async, [&msg, &cloud] { ufomap::toUfomap(msg, cloud); });

        ufomap_math::Pose6 transform = ufomap::toUfomap(a1.get().transform);
        a2.wait();
        cloud.transform(transform);
        if (insert_discrete_)
        {
            map_.insertPointCloudDiscrete(transform.translation(), cloud, max_range_, insert_n_,
                                                                        insert_depth_);
        }
        else
        {
            map_.insertPointCloud(transform.translation(), cloud, max_range_);
        }

        if (clear_robot_enabled_)
        {
            ufomap::Point3 robot_bbx_min(transform.x() - robot_radius_,
                                                                     transform.y() - robot_radius_,
                                                                     transform.z() - (robot_height_ / 2.0));
            ufomap::Point3 robot_bbx_max(transform.x() + robot_radius_,
                                                                     transform.y() + robot_radius_,
                                                                     transform.z() + (robot_height_ / 2.0));
            map_.clearAreaBBX(robot_bbx_min, robot_bbx_max, 0);
        }
    }
    catch (tf2::TransformException& ex)
    {
        ROS_WARN_THROTTLE(1, "%s", ex.what());
    }
}

void BathyMapper::timerCallback(const ros::TimerEvent&)
{
    std_msgs::Header header;
    header.stamp = ros::Time::now();
    header.frame_id = frame_id_;

    if (0 < map_pub_.getNumSubscribers() || map_pub_.isLatched())
    {
        ufomap_msgs::Ufomap msg;
        ufomap_msgs::mapToMsg(map_, msg, false);
        msg.header = header;
        map_pub_.publish(msg);
    }

    if (0 < map_binary_pub_.getNumSubscribers() || map_binary_pub_.isLatched())
    {
        ufomap_msgs::Ufomap msg;
        ufomap_msgs::mapToMsg(map_, msg, false, true);
        msg.header = header;
        map_binary_pub_.publish(msg);
    }

    if (0 < cloud_pub_.getNumSubscribers() || cloud_pub_.isLatched())
    {
        ufomap::PointCloud cloud;
        for (auto it = map_.begin_leafs(true, false, false, false, 0),
                            it_end = map_.end_leafs();
                 it != it_end; ++it)
        {
            cloud.push_back(it.getCenter());
        }
        sensor_msgs::PointCloud2::Ptr cloud_msg(new sensor_msgs::PointCloud2);
        ufomap::fromUfomap(cloud, cloud_msg);
        cloud_msg->header = header;
        cloud_pub_.publish(cloud_msg);
    }
}

void BathyMapper::configCallback(bathy_mapper::ServerConfig& config, uint32_t level)
{
    frame_id_ = config.frame_id;
    max_range_ = config.max_range;
    insert_discrete_ = config.insert_discrete;
    insert_depth_ = config.insert_depth;
    insert_n_ = config.insert_n;
    clear_robot_enabled_ = config.clear_robot;
    robot_height_ = config.robot_height;
    robot_radius_ = config.robot_radius;

    if (pub_rate_ != config.pub_rate)
    {
        pub_rate_ = config.pub_rate;
        if (0 < pub_rate_)
        {
            pub_timer_ =
                    nh_priv_.createTimer(ros::Rate(pub_rate_), &BathyMapper::timerCallback, this);
        }
    }

    transform_timeout_.fromSec(config.transform_timeout);

    if (cloud_in_queue_size_ != config.cloud_in_queue_size)
    {
        cloud_sub_ = nh_.subscribe("cloud_in", config.cloud_in_queue_size,
                                                             &BathyMapper::cloudCallback, this);
    }

    if (map_pub_.isLatched() != config.map_latch ||
            map_queue_size_ != config.map_queue_size)
    {
        map_pub_ = nh_priv_.advertise<ufomap_msgs::Ufomap>("map", config.map_queue_size,
                                                                                                             config.map_latch);
    }

    if (map_binary_pub_.isLatched() != config.map_binary_latch ||
            map_binary_queue_size_ != config.map_binary_queue_size)
    {
        map_binary_pub_ = nh_priv_.advertise<ufomap_msgs::Ufomap>(
                "map_binary", config.map_binary_queue_size, config.map_binary_latch);
    }

    if (cloud_pub_.isLatched() != config.map_cloud_latch ||
            map_cloud_queue_size_ != config.map_cloud_queue_size)
    {
        cloud_pub_ = nh_priv_.advertise<sensor_msgs::PointCloud2>(
                "map_cloud", config.map_cloud_queue_size, config.map_cloud_latch);
    }
}

void BathyMapper::loadMap(std::string& filename){
    map_.read(filename);
}

void BathyMapper::saveMap(std::string& filename){
    map_.write(filename);
}

}  // namespace bathy_mapper
