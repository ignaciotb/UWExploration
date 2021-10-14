#include <bathy_graph_slam/serial.hpp>

int main(int argc, char **argv)
{

    // gtsam::Key i1 = gtsam::symbol('x', 1);
    // gtsam::Key i2 = gtsam::symbol('x', 2);
    // gtsam::Key i3 = gtsam::symbol('x', 3);
    // gtsam::Key j1 = gtsam::symbol('l', 1);
    // gtsam::Key j2 = gtsam::symbol('l', 2);

    // gtsam::NonlinearFactorGraph graph;
    // gtsam::Pose2 priorMean = gtsam::Pose2(0.0, 0.0, 0.0);
    // gtsam::Vector sigma_p(3);
    // sigma_p << 0.3, 0.3, 0.1;
    // gtsam::SharedNoiseModel priorNoise = gtsam::noiseModel::Diagonal::Sigmas(sigma_p);
    // PriorFactorPose2 prior = PriorFactorPose2(i1, priorMean, priorNoise);
    // graph.add(prior);

    // gtsam::Pose2 odometry = gtsam::Pose2(2.0, 0.0, 0.0);
    // gtsam::Vector sigma(3);
    // sigma << 0.2, 0.2, 0.2;
    // gtsam::SharedNoiseModel odometryNoise = gtsam::noiseModel::Diagonal::Sigmas(sigma);

    // graph.add(BetweenFactorPose2(i1, i2, odometry, odometryNoise));
    // graph.add(BetweenFactorPose2(i2, i3, odometry, odometryNoise));

    // Values initialEstimate;
    // initialEstimate.insert(i1, Pose2(0.5, 0.0, 0.2));
    // initialEstimate.insert(i2, Pose2(2.3, 0.1, -0.2));
    // initialEstimate.insert(i3, Pose2(4.1, 0.1, 0.1));
    // initialEstimate.insert(j1, Point2(1.8, 2.1));
    // initialEstimate.insert(j2, Point2(4.1, 1.8));

    // std::string serialized;
    // serialized = gtsam::serialize(graph);
    // std::cout << serialized;
    
    // Test serialization
    gtsam::NonlinearFactorGraph graph;
    Values initialEstimate;
    std::string path_graph = "/home/torroba/catkin_workspaces/auv_ws/src/UWExploration/utils/uw_tests/slam/overnight_2020/graph_init";
    std::string OutputGraphFile(path_graph + "_serial.txt");
    std::string OutputValsFile(path_graph + "_values_serial.txt");

    // // Test saving in disk
    // std::ofstream OfsOutput(OutputGraphFile.c_str());
    // boost::archive::text_oarchive OaOutput(OfsOutput);
    // OaOutput << graph;
    // OfsOutput.close();

    // std::ofstream OfsValsOutput(OutputValsFile.c_str());
    // boost::archive::text_oarchive OaValsOutput(OfsValsOutput);
    // OaValsOutput << initialEstimate;
    // OfsValsOutput.close();

    // Test reading from disk
    std::ifstream IfsInput(OutputGraphFile.c_str());
    boost::archive::text_iarchive IaInput(IfsInput);
    gtsam::NonlinearFactorGraph DeserializedGraph;
    IaInput >> DeserializedGraph;

    cout << "Deserialized factor graph: " << endl;
    DeserializedGraph.print();
    IfsInput.close();

    std::ifstream IfsValsInput(OutputValsFile.c_str());
    boost::archive::text_iarchive IaValsInput(IfsValsInput);
    Values DeserializedVals;
    IaValsInput >> DeserializedVals;

    cout << "Deserialized estimated values: " << endl;
    DeserializedVals.print();
    IfsValsInput.close();

    return 0;
}