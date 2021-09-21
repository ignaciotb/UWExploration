#include <bathy_graph_slam/sam_graph.hpp>

samGraph::samGraph()
{
    // ISAM solver params
    ISAM2Params params = ISAM2Params(ISAM2GaussNewtonParams(0.001), 0.0,
                                     0, false, true,
                                     ISAM2Params::CHOLESKY, true,
                                     DefaultKeyFormatter, true);

    // TODO: this has to be an input param
    Vector odoSigmas = Vector3(0.005, 0.001, 0.01);
    odoNoise = noiseModel::Diagonal::Sigmas(odoSigmas);

    // Instantiate pointers
    isam_.reset(new ISAM2(params));
    newFactors_.reset(new NonlinearFactorGraph());
    initValues_.reset(new Values());
}

samGraph::~samGraph(){
}

void samGraph::addPrior()
{
    SharedDiagonal odoNoise = noiseModel::Diagonal::Sigmas(
        (Vector(3) << 0.1, 0.1, M_PI / 100.0).finished());

    ISAM2Params params = ISAM2Params(ISAM2GaussNewtonParams(0.001), 0.0,
                                     0, false, true,
                                     ISAM2Params::CHOLESKY, true,
                                     DefaultKeyFormatter, true);

    ISAM2 isam(params);
    Values fullinit;
    NonlinearFactorGraph fullgraph;

    // i keeps track of the time step
    size_t i = 0;

    // Add a prior at time 0 and update isam
    {
        NonlinearFactorGraph newfactors;
        newfactors.addPrior(0, Pose2(0.0, 0.0, 0.0), odoNoise);
        fullgraph.push_back(newfactors);

        Values init;
        init.insert((0), Pose2(0.01, 0.01, 0.01));
        fullinit.insert((0), Pose2(0.01, 0.01, 0.01));

        isam.update(newfactors, init);
    }
    ROS_INFO("Prior updated");
    // Add odometry from time 0 to time 5
    for (; i < 5; ++i)
    {
        NonlinearFactorGraph newfactors;
        newfactors += BetweenFactor<Pose2>(i, i + 1, Pose2(1.0, 0.0, 0.0), odoNoise);
        fullgraph.push_back(newfactors);

        Values init;
        init.insert((i + 1), Pose2(double(i + 1) + 0.1, -0.1, 0.01));
        fullinit.insert((i + 1), Pose2(double(i + 1) + 0.1, -0.1, 0.01));
        ROS_INFO("About to update ISAM");
        isam.update(newfactors, init);
        ROS_INFO("ISAM updated");
    }
}

void samGraph::addOdomFactor(Pose2 odom_step, size_t step)
{
    // // Add odom factor
    // ROS_INFO("Adding odom factor");
    // NonlinearFactorGraph newfactors;
    // newfactors += BetweenFactor<Pose2>(step, step + 1, Pose2(1.0, 0.0, 0.0), odoNoise);
    // // newFactors_->push_back(BetweenFactor<Pose2>(step - 1, step, odom_step, odoNoise));
    // std::cout << "New factor created" << std::endl;

    // // predict pose and add as initial estimate
    // Values init;
    // // Pose2 predictedPose = lastPose_.compose(odom_step);
    // // lastPose_ = predictedPose;
    // init.insert((step + 1), Pose2(double(step + 1) + 0.1, -0.1, 0.01));
    // std::cout << "New init created" << std::endl;

    // isam_->update(newfactors, init);
    // std::cout << "adding odom factor" << std::endl;
    size_t i = 0;
    ROS_INFO("Odom factors added");

    for (; i < 5; ++i)
    {
        NonlinearFactorGraph newfactors;
        newfactors += BetweenFactor<Pose2>(i, i + 1, Pose2(1.0, 0.0, 0.0), odoNoise);
        // std::cout << newfactors.print() << std::endl;
        // fullgraph.push_back(newfactors);

        Values init;
        init.insert((i + 1), Pose2(double(i + 1) + 0.1, -0.1, 0.01));
        // fullinit.insert((i + 1), Pose2(double(i + 1) + 0.1, -0.1, 0.01));

        isam_->update(newfactors, init);
    }
    ROS_INFO("ISAM updated again");
}
// void samGraph::initializeISAM()
// {
//     initValues_->print();
//     newFactors_->print();
//     LevenbergMarquardtOptimizer batchOptimizer(*newFactors_, *initValues_);
//     Values result = batchOptimizer.optimize();
//     // isam_->update(*newFactors_, result);

//     // initialized = true;
// }
