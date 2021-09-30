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

    // Landmark ids of previous submap
    lm_idx_prev_.clear();

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

bool samGraph::addLandmarksFactor(PointCloudT& landmarks, size_t step, 
                                  std::vector<int>& lm_idx, Pose3 submap_pose)
{
    int i = 0;
    bool lc_detected = false;
    for (PointT &point : landmarks){
        // If known association is on, the landmarks are already in map frame
        Point3 lm = Eigen::Vector3f(point.x, point.y, point.z).cast<double>();
        
        // Add 3D Bearing-range factor for every landmark        
        graph_->emplace_shared<BearingRangeFactor<Pose3, Point3> >(
            Symbol('x', step+1), Symbol('l', lm_idx.at(i)), 
            submap_pose.bearing(lm), submap_pose.range(lm), brNoise_);

        // Add initial estimate for landmarks unless already added (LC)
        if (!initValues_->exists(Symbol('l', lm_idx.at(i)))) {
            // Point3 mapLandmark = submap_pose.transformFrom(lm);
            initValues_->insert(Symbol('l', lm_idx.at(i)), lm);
        }
        else{
            // Check that the landmark revisited doesn't belong to the previous submap
            auto prev_lm = lm_idx_prev_.begin();
            prev_lm = std::find_if(prev_lm, lm_idx_prev_.end(), [&](const int prev_lm_id) {
                return prev_lm_id == lm_idx.at(i);
            });
            // If it doesn't belong to the previous submap, it's a new LC
            if(prev_lm == lm_idx_prev_.end()){
                lc_detected = true;
                // std::cout << "LC with landmark " << lm_idx.at(i) << std::endl;
            }
        }
        i++;
    }

    // Store latest landmarks for next round
    lm_idx_prev_.clear();
    lm_idx_prev_ = lm_idx;
    std::cout << "RB factors added" << std::endl;

    return lc_detected;
} 

void samGraph::updateISAM2(int iterations)
{
    isam_->update(*graph_, *initValues_);

    // Each call to iSAM2 update(*) performs one iteration of the iterative nonlinear solver.
    // If accuracy is desired at the expense of time, update(*) can be called additional
    // times to perform multiple optimizer iterations every step.
    if(iterations > 1){
        for(int i=0; i<iterations-1; i++){
            isam_->update();
        }
    }

    // initValues_->print("Init estimate: ");
    graph_.reset(new NonlinearFactorGraph());
    initValues_.reset(new Values());
    std::cout << "ISAM updated" << std::endl;
}

Values samGraph::computeEstimate()
{
    Values estimate = isam_->calculateEstimate();
    estimate.print("Estimated values: ");

    return estimate;
}


void samGraph::saveResults(const Values &result, const std::string &outfilename)
{
    std::fstream fs(outfilename.c_str(), std::fstream::out);

    auto index = [](gtsam::Key key){ return Symbol(key).index(); };

    // save 3D poses
    for (const auto key_value : result)
    {
        auto p = dynamic_cast<const GenericValue<Pose3> *>(&key_value.value);
        if (!p)
            continue;
        const Pose3 &pose = p->value();
        Vector quaternion = Rot3(pose.rotation()).quaternion();
        fs << "Pose " << index(key_value.key) << " " << pose.x() << " "
            << pose.y() << " " << pose.z() << " " << quaternion(3) 
            << " " << quaternion(2) << " " << quaternion(1) << " " << quaternion(0) 
            << std::endl;
    }

    // save 3D landmarks
    for (const auto key_value : result)
    {
        auto p = dynamic_cast<const GenericValue<Point3> *>(&key_value.value);
        if (!p)
            continue;
        const Point3 &point = p->value();
        fs << "Landmark " << index(key_value.key) << " " << point.x() << " "
            << point.y() << " " << point.z() << std::endl;
    }

    fs.close();
    std::cout << "Results saved" << std::endl;
}