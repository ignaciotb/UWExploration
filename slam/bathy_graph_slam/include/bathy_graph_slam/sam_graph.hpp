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

    samGraph();

    ~samGraph();

    void addPrior();

    void addOdomFactor(Pose2 odom_step, size_t step);

    // void initializeISAM();

    // Create a factor graph
    NonlinearFactorGraph::shared_ptr newFactors_;
    Values::shared_ptr initValues_;
    boost::shared_ptr<ISAM2> isam_;
    Pose2 lastPose_;

    // TODO: this has to be an input parameter
    SharedDiagonal odoNoise; 
};


#endif