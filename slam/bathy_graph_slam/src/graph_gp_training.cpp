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
#include <actionlib/server/simple_action_server.h>
#include <svgp_mapping/batch_trainingAction.h>

#include <bathy_graph_slam/serial.hpp>

using namespace std;
using namespace gtsam;

class GraphGPTraining
{

private:
    std::string node_name_;
    ros::NodeHandle *nh_;
    actionlib::SimpleActionServer<svgp_mapping::batch_trainingAction> as_;
    svgp_mapping::batch_trainingResult a_result_;
    Marginals *marginals_;
    KeyVector landmarksKeys_;
    std::vector<Point3> landmarksPoses_;

public:
    GraphGPTraining(std::string node_name, ros::NodeHandle &nh) : node_name_(node_name), nh_(&nh),
                                                               as_(*nh_, node_name_, 
                                                               boost::bind(&GraphGPTraining::executeCB, this, _1), false)
    {
        // Start action server
        as_.start();

        std::string path_graph, init_file, out_file, graph_as_name;
        nh_->param<std::string>("graph_init_file", path_graph, "graph_init");
        nh_->param<std::string>("results_file", out_file, "results.txt");
        nh_->param<std::string>("initial_file", init_file, "initial.txt");
        // nh_->param<std::string>("graph_as", graph_as_name, "graph_as");

        // Create a factor graph and load dataset
        NonlinearFactorGraph graph;
        Values estimate;

        // Test reading from disk
        std::string OutputGraphFile(path_graph + "_serial.txt");
        std::ifstream IfsInput(OutputGraphFile.c_str());
        boost::archive::text_iarchive IaInput(IfsInput);
        IaInput >> graph;
        cout << "Deserialized factor graph: " << endl;
        // graph.print();
        IfsInput.close();

        std::string OutputValsFile(path_graph + "_values_serial.txt");
        std::ifstream IfsValsInput(OutputValsFile.c_str());
        boost::archive::text_iarchive IaValsInput(IfsValsInput);
        IaValsInput >> estimate;
        cout << "Deserialized estimated values: " << endl;
        // estimate.print();
        IfsValsInput.close();

        // marginals_ = new Marginals(graph, estimate);
        std::cout << "Marginals computed " << std::endl;

        // Save initial estimate for plotting
        // saveResults(estimate, init_file);

        // // Optimize
        Values result = solverLM(graph, estimate);
        // // Values result = this->solverISAM(*graph, *initialEstimate);
        // ROS_INFO("%s: SLAM solved ", node_name_.c_str());

        // // Get keys for map landmarks
        this->getLMKeys(result);

        // Compute solution joint marginals
        marginals_ = new Marginals(graph, result);
        // std::cout << "--------------" << std::endl;
        // marginals_->print();

        // Save trajectory and map
        // saveResults(result, out_file);

        ros::spin();
    }

    ~GraphGPTraining()
    {
    }

    void executeCB(const svgp_mapping::batch_trainingGoalConstPtr &goal)
    {
        bool success = true;
        ROS_INFO("-----------Request received------------");

        // Parse goal into KeyVector lm
        KeyVector lm;
        std::vector<Point3> lmPoses;
        for (int i = 0; i < goal->keys.size(); i++)
        {
            lm.push_back(landmarksKeys_.at(goal->keys.at(i)));
            // std::cout << "Landmark key " << landmarksKeys_.at(goal->keys.at(i)) << std::endl;
            lmPoses.push_back(landmarksPoses_.at(goal->keys.at(i)));
        }
        // std::cout << "Landmarks requested " << lmPoses.size() << std::endl;

        // Key val = Symbol('l', 223686);
        // Matrix joint = marginals_->marginalCovariance(val);
        // std::cout << joint.matrix() << std::endl;

        JointMarginal joint = marginals_->jointMarginalCovariance(lm);
        std::cout << "Marginals computed " << std::endl;

        // std::cout << joint.fullMatrix() << std::endl;
        // std::cout << "Number of landmarks requested " << lm.size() << std::endl;

        if (success)
        {
            // Result containing the landmarks vector and its upper triangular cov matrix
            std::vector<double> lm_vec;
            for (int i = 0; i < lmPoses.size(); i++)
            {
                for (int j = 0; j < 3; j++)
                { // TODO: make size-agnostic
                    lm_vec.push_back(lmPoses.at(i)[j]);
                }
            }
            a_result_.lm_vec = lm_vec;

            // Store upper triangular half of cov matrix
            std::vector<double> cov_vec;
            Eigen::MatrixXd joint_cov = joint.fullMatrix();
            // std::cout << joint.fullMatrix() << std::endl;
            int rows = joint_cov.rows();
            int cols = joint_cov.cols();
            int cnt = 0;
            for (int i = 0; i < rows; i++)
            {
                for (int j = cnt; j < cols; j++)
                {
                    cov_vec.push_back(joint_cov(i, j));
                }
                cnt++;
            }
            a_result_.ut_cov_mat = cov_vec;

            ROS_DEBUG("%s: Succeeded", node_name_.c_str());
            as_.setSucceeded(a_result_);
        }
    }

    void saveResults(const Values &result, const string &outfilename)
    {
        fstream stream(outfilename.c_str(), fstream::out);

        auto index = [](gtsam::Key key)
        { return Symbol(key).index(); };

        // save 2D poses
        for (const auto key_value : result)
        {
            auto p = dynamic_cast<const GenericValue<Pose2> *>(&key_value.value);
            if (!p)
                continue;
            const Pose2 &pose = p->value();
            stream << "Pose " << index(key_value.key) << " " << pose.x() << " "
                   << pose.y() << " " << pose.theta() << endl;
        }

        // save 2D landmarks
        for (const auto key_value : result)
        {
            auto p = dynamic_cast<const GenericValue<Point3> *>(&key_value.value);
            if (!p)
                continue;
            const Point3 &point = p->value();
            stream << "Landmark " << index(key_value.key) << " " << point.x() << " "
                   << point.y() << point.z() << endl;
        }

        stream.close();
        std::cout << "Results saved" << std::endl;
    }

    void getLMKeys(const Values &result)
    {
        auto index = [](gtsam::Key key)
        { return Symbol(key).index(); };

        for (const auto key_value : result)
        {
            auto p = dynamic_cast<const GenericValue<Point3> *>(&key_value.value);
            if (!p)
                continue;

            const Point3 &point = p->value();
            landmarksPoses_.push_back(point);
            landmarksKeys_.push_back(Symbol('l', index(key_value.key)));
            // std::cout << "Landmark key index " << index(key_value.key) << std::endl;
        }
        std::cout << "Landmarks poses " << landmarksPoses_.size() << std::endl;
    }

    Values solverLM(NonlinearFactorGraph &graph, Values &initialEstimate)
    {
        // Optimize
        LevenbergMarquardtParams paramsLM;
        //   paramsLM.linearSolverType = LevenbergMarquardtParams::MULTIFRONTAL_QR;
        LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);

        return optimizer.optimize();
    }

    Values solverISAM2(NonlinearFactorGraph &graph, Values &initialEstimate)
    {
        ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.0001;
        // parameters.relinearizeSkip = 1;
        ISAM2 isam(parameters);

        isam.update(graph, initialEstimate);

        return isam.calculateEstimate();
    }

    Values solverISAM(NonlinearFactorGraph &graph, Values &initialEstimate)
    {
        int relinearizeInterval = 1;
        NonlinearISAM isam(relinearizeInterval);

        isam.update(graph, initialEstimate);

        return isam.estimate();
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "graph_gp_training");
    ros::NodeHandle nh("~");
    GraphGPTraining *graph_gp_training = new GraphGPTraining(ros::this_node::getName(), nh);

    ros::waitForShutdown();
    if (!ros::ok())
    {
        delete graph_gp_training;
    }
    ROS_INFO("Finishing Graph GP training");
    return 0;
}
