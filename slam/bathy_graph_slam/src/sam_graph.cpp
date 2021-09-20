#include <bathy_graph_slam/sam_graph.hpp>

samGraph::samGraph()
{
    // ISAM solver params
    ISAM2Params params = ISAM2Params(ISAM2GaussNewtonParams(0.001), 0.0,
                                     0, false, true,
                                     ISAM2Params::CHOLESKY, true,
                                     DefaultKeyFormatter, true);

    // TODO: this has to be an input param
    odoNoise = noiseModel::Diagonal::Sigmas((Vector(3) << 0.1, 0.1, M_PI / 100.0).finished());

    // Instantiate pointers
    isam_.reset(new ISAM2(params));
    newFactors_.reset(new NonlinearFactorGraph());
    initValues_.reset(new Values());
}

samGraph::~samGraph(){
}

void samGraph::addPrior()
{
    // Add prior to graph
    NonlinearFactorGraph factors;
    Pose2 initPose(0.0, 0.0, 0.0);
    factors.addPrior(0, initPose, odoNoise);
    // newFactors_->addPrior(0, initPose, odoNoise);
    newFactors_->push_back(factors);

    // First pose to initial estimate
    Pose2 pose_init_corrupted(0.01, 0.01, 0.01);
    initValues_->insert((0), pose_init_corrupted);

    // Initialize last pose with init one
    lastPose_ = initPose;
    std::cout << "Prior added" << std::endl;

    // isam_->update(newFactors_, init);
}

void samGraph::addOdomFactor(Pose2 odom_step, size_t step)
{
    // Add odom factor
    newFactors_->push_back(BetweenFactor<Pose2>(step - 1, step, odom_step, odoNoise));

    // predict pose and add as initial estimate
    Pose2 predictedPose = lastPose_.compose(odom_step);
    lastPose_ = predictedPose;
    initValues_->insert(step, predictedPose);
}

