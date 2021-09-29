#include <bathy_graph_slam/sam_graph.hpp>

samGraph::samGraph()
{
    // ISAM solver params
    ISAM2Params params = ISAM2Params(ISAM2GaussNewtonParams(0.001), 0.0,
                                     0, false, true,
                                     ISAM2Params::CHOLESKY, true,
                                     DefaultKeyFormatter, true);

    num_landmarks_ = 0;

    // TODO: this has to be an input param
    odoNoise_ = noiseModel::Diagonal::Sigmas((Vector(6) << 0.1, 0.1, 0,1,
                                            0, 0, M_PI / 100.0).finished());
    brNoise_ = noiseModel::Diagonal::Sigmas((Vector(3)<<0.01,0.03,0.05).finished());
    // Instantiate pointers
    isam_.reset(new ISAM2(params));
    graph_.reset(new NonlinearFactorGraph());
    initValues_.reset(new Values());
}

samGraph::~samGraph()
{
}

void samGraph::addPrior(Pose3& initPose)
{
    // Add a prior at time 0 and update isam
    graph_->addPrior(Symbol('x', 0), initPose, odoNoise_);
    initValues_->insert(Symbol('x', 0), initPose);
    // Init last pose where the odom frame is
    lastPose_ = initPose;

    // TODO: do we need to update isam here?
    // isam_->update(*graph_, *initValues_);
    // graph_.reset(new NonlinearFactorGraph());
    // initValues_.reset(new Values());
    std::cerr << "Prior added " << std::endl;
}

void samGraph::addOdomFactor(Pose3 factor_pose, Pose3 odom_step, size_t step)
{
    // Add odometry
    // submap i will be DR factor i+1 since the origin 
    // (where there's no submap) is the factor 0
    graph_->emplace_shared<BetweenFactor<Pose3> >(Symbol('x', step), Symbol('x', step+1),
                                                  odom_step, odoNoise_);

    // Predict pose and add as initial estimate
    // Pose3 predictedPose = lastPose_.compose(odom_step);
    // lastPose_ = predictedPose;
    // predictedPose.print("Node added ");
    initValues_->insert(Symbol('x', step+1), factor_pose);
    std::cerr << "Odom factor added" << std::endl;

}

void samGraph::addLandmarksFactor(PointCloudT& landmarks, size_t step, 
                                  std::vector<int>& lm_idx, Pose3 submap_pose)
{
    // Check we've got the same number of landmarks and indexes
    if(!landmarks.points.size() == lm_idx.size()){
        std::cerr << "Different number of landmarks and indexes" << std::endl;
    }

    // Convert landmarks PCL points to gtsam Point3
    int i = 0;
    bool lc_detected = false;
    for (PointT &point : landmarks){
        // If known association is on, the landmarks are already in map frame
        Point3 lm = Eigen::Vector3f(point.x, point.y, point.z).cast<double>();
        
        graph_->emplace_shared<BearingRangeFactor<Pose3, Point3> >(
            Symbol('x', step+1), Symbol('l', lm_idx.at(i)), 
            submap_pose.bearing(lm), submap_pose.range(lm), brNoise_);

        // Add initial estimate for landmarks
        if (!initValues_->exists(Symbol('l', lm_idx.at(i)))) {
            // Point3 mapLandmark = submap_pose.transformFrom(lm);
            initValues_->insert(Symbol('l', lm_idx.at(i)), lm);
        }
        else{
            lc_detected = true;
            // std::cout << "LC with landmark " << lm_idx.at(i) << std::endl;
            // TODO: if loop closure detected, updateISAM2()
        }
        i++;
    }
    if(lc_detected){
        std::cout << "Loop closure detected" << std::endl;
    }

    // initValues_->print("Init values ");
    std::cout << "RB factors added" << std::endl;
} 

void samGraph::updateISAM2()
{
    isam_->update(*graph_, *initValues_);

    // Values estimate = isam_->calculateEstimate();
    initValues_->print("Init estimate: ");
    graph_.reset(new NonlinearFactorGraph());
    initValues_.reset(new Values());
    std::cout << "ISAM updated" << std::endl;
}
