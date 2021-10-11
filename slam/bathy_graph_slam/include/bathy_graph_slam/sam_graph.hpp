#ifndef SAM_GRAPH_HPP
#define SAM_GRAPH_HPP

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Point3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/geometry/BearingRange.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearISAM.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <fstream>      // std::fstream

using namespace gtsam;

typedef pcl::PointCloud<pcl::PointXYZ> PointCloudT;
typedef pcl::PointXYZ PointT;

class samGraph
{
    typedef BearingRangeFactor<Pose3, Point3> BearingRangeFactor3D;

public:
    samGraph();

    ~samGraph();

    void addPrior(Pose3& initPose);

    void addOdomFactor(Pose3 factor_pose, Pose3 odom_step, size_t step);

    bool addLandmarksFactor(PointCloudT&, size_t step, 
                            std::vector<int>& lm_idx, Pose3 submap_pose);

    void updateISAM2(int iterations);

    Values computeEstimate();

    void saveResults(const Values &result, const std::string &outfilename);


    // Create a factor graph
    NonlinearFactorGraph::shared_ptr graph_;
    Values::shared_ptr initValues_;
    boost::shared_ptr<ISAM2> isam_;
    Pose3 lastPose_;
    // Keep track of landmarks observed
    int num_landmarks_;
    // std::vector<int> lm_idx_vec_;
    // std::vector<int> lm_mapped_idx_vec_;
    std::vector<int> lm_idx_prev_;

    // TODO: this has to be an input parameter
    SharedDiagonal odoNoise_;
    SharedDiagonal brNoise_;
};

#endif