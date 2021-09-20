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

class samGraph{
    public:

    samGraph(){

    }
};


#endif