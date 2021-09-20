#ifndef SAM_GRAPH_HPP
#define SAM_GRAPH_HPP

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Point2.h>

#include <gtsam/inference/Symbol.h>

#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/sam/BearingRangeFactor.h>

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/slam/dataset.h>
#include <gtsam/nonlinear/Values.h>

#include <gtsam/nonlinear/NonlinearISAM.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <ros/ros.h>

using namespace gtsam;

class samGraph{
    public:

    samGraph(){

        // ISAM solver params
        ISAM2Params params = ISAM2Params(ISAM2GaussNewtonParams(0.001), 0.0,
                                          0, false, true,
                                          ISAM2Params::CHOLESKY, true,
                                          DefaultKeyFormatter, true);

        odoNoise = noiseModel::Diagonal::Sigmas((Vector(3) << 0.1, 0.1, M_PI / 100.0).finished());

        // Instantiate pointers
        isam = new ISAM2(params);
        fullGraph.reset(new NonlinearFactorGraph());
        fullInit.reset(new Values());
    }

    void addPrior(){
        NonlinearFactorGraph newfactors;
        Pose2 pose_init(0.0, 0.0, 0.0);
        newfactors.addPrior(0, pose_init, odoNoise);
        fullGraph->push_back(newfactors);

        Values init;
        Pose2 pose_init_corrupted(0.01, 0.01, 0.01);
        init.insert((0), pose_init_corrupted);
        fullInit->insert((0), pose_init_corrupted);

        isam->update(newfactors, init);
    }

    // Create a factor graph
    NonlinearFactorGraph::shared_ptr fullGraph;
    Values::shared_ptr fullInit;
    ISAM2* isam;

    // TODO: this has to be an input parameter
    SharedDiagonal odoNoise; 
};


#endif