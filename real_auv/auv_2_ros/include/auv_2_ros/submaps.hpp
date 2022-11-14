#ifndef SUBMAPS_HPP
#define SUBMAPS_HPP

#include <fstream>
#include <iostream>
#include <regex>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

#include <Eigen/Core>

#include "data_tools/std_data.h"
#include <data_tools/navi_data.h>
#include <data_tools/transforms.h>

using namespace std;
using namespace Eigen;
typedef std::vector<Eigen::MatrixXd, Eigen::aligned_allocator<Eigen::MatrixXd>> PointsT;
typedef pcl::PointCloud<pcl::PointXYZ> PointCloudT;
typedef pcl::PointXYZ PointT;
typedef std::vector<Vector3d, aligned_allocator<Vector3d>> corners;
typedef std::vector<Eigen::Matrix2d, Eigen::aligned_allocator<Eigen::Matrix2d> > covs;

class SubmapObj{

private:

public:

    int submap_id_;
    int swath_id_;
    PointCloudT submap_pcl_;
    std::vector<int> overlaps_idx_;
    Eigen::Vector3d colors_;
    Eigen::Isometry3f submap_tf_;
    Eigen::Matrix<double,6,6> submap_info_;
    Eigen::Matrix<double,6,6> submap_lc_info_;
    Eigen::MatrixXd auv_tracks_;

    SubmapObj();

    SubmapObj(const unsigned int& submap_id, const unsigned int& swath_id, PointCloudT& submap_pcl);

    void findOverlaps(bool submaps_in_map_tf, std::vector<SubmapObj, Eigen::aligned_allocator<SubmapObj> > &submaps_set);

    Eigen::Matrix<double, 6, 6> createDRWeights();

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class MapObj: public SubmapObj{

public:

    MapObj();

    MapObj(PointCloudT& map_pcl);

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

template<class Archive>
void save(Archive & archive,
          SubmapObj const & m)
{
    Eigen::MatrixXf points = m.submap_pcl_.getMatrixXfMap(3,4,0).transpose();
    archive(CEREAL_NVP(m.submap_id_), CEREAL_NVP(m.swath_id_), CEREAL_NVP(points),
        CEREAL_NVP(m.overlaps_idx_), CEREAL_NVP(m.colors_), CEREAL_NVP(m.submap_tf_.matrix()), CEREAL_NVP(m.submap_info_),
        CEREAL_NVP(m.auv_tracks_));
}

template<class Archive>
void load(Archive & archive,
          SubmapObj & m)
{
    Eigen::MatrixXf points;
    archive(CEREAL_NVP(m.submap_id_), CEREAL_NVP(m.swath_id_), CEREAL_NVP(points),
        CEREAL_NVP(m.overlaps_idx_), CEREAL_NVP(m.colors_), CEREAL_NVP(m.submap_tf_.matrix()), CEREAL_NVP(m.submap_info_),
        CEREAL_NVP(m.auv_tracks_));
    for(unsigned int i=0; i<points.rows(); i++){
        m.submap_pcl_.points.push_back(PointT(points.row(i)[0], points.row(i)[1], points.row(i)[2]));
    }
}

typedef std::vector<SubmapObj, Eigen::aligned_allocator<SubmapObj>> SubmapsVec;

void readSubmapFile(const string submap_str, PointCloudT::Ptr submap_pcl);

std::vector<std::string> checkFilesInDir(DIR *dir);

std::vector<SubmapObj, Eigen::aligned_allocator<SubmapObj> > readSubmapsInDir(const string& dir_path);

Array3f computeInfoInSubmap(const SubmapObj& submap);

SubmapsVec parseSubmapsAUVlib(std_data::pt_submaps& ss);

std::tuple<MapObj, Isometry3d> parseMapAUVlib(std_data::pt_submaps& ss);

SubmapsVec parsePingsAUVlib(std_data::mbes_ping::PingsT& pings);

SubmapsVec createSubmaps(SubmapsVec& pings, int submap_size);

SubmapsVec createMap(SubmapsVec& pings, int submap_size);

void transformSubmapObj(SubmapObj& submap, Isometry3f& poseDRt);

std::pair<int, corners> getSubmapCorners(bool submaps_in_map_tf, const SubmapObj& submap);

bool checkSubmapsOverlap(const corners submap_i_corners, const corners submap_k_corners);

bool pointToLine(const Vector3d seg_a, const Vector3d seg_b, const Vector3d point_c);

bool checkSubmapSize(const SubmapObj& submap_i);

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

PointsT pclToMatrixSubmap(const SubmapsVec& submaps_set);

PointsT trackToMatrixSubmap(const SubmapsVec& submaps_set);

std::pair<Eigen::Matrix2d, Eigen::Matrix2d> readCovMatrix(const std::string& file_name);

covs readCovsFromFiles(boost::filesystem::path folder);

std::vector<std::vector<int> > readGTLoopClosures(string& fileName, int submaps_nb);

#endif // SUBMAPS_HPP
