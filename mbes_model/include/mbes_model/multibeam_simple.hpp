#ifndef MULTIBEAM_SIMPLE_HPP
#define MULTIBEAM_SIMPLE_HPP

#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include "submaps_tools/submaps.hpp"

template <typename PointT>
class MultibeamSensor: public pcl::VoxelGrid<PointT>
{
protected:

    using pcl::VoxelGrid<PointT>::min_b_;
    using pcl::VoxelGrid<PointT>::max_b_;
    using pcl::VoxelGrid<PointT>::div_b_;
    using pcl::VoxelGrid<PointT>::leaf_size_;
    using pcl::VoxelGrid<PointT>::inverse_leaf_size_;

    using PointCloudFilt = typename pcl::Filter<PointT>::PointCloud;
    using PointCloudFiltPtr = typename PointCloudFilt::Ptr;
    using PointCloudFiltConstPtr = typename PointCloudFilt::ConstPtr;

public:
  /** \brief Empty constructor. */
  MultibeamSensor ()
  {
    initialized_ = false;
    this->setSaveLeafLayout (true);
  }

  /** \brief Destructor. */
  ~MultibeamSensor ()
  {
  }

  inline PointCloudFilt
  getFilteredPointCloud () { return filtered_cloud_; }


  /** \brief Returns the minimum bounding of coordinates of the voxel grid (x,y,z).
    * \return the minimum coordinates (x,y,z)
    */
  inline Eigen::Vector3f
  getMinBoundCoordinates () { return (b_min_.head<3> ()); }

  /** \brief Returns the maximum bounding of coordinates of the voxel grid (x,y,z).
    * \return the maximum coordinates (x,y,z)
    */
  inline Eigen::Vector3f
  getMaxBoundCoordinates () { return (b_max_.head<3> ()); }

  /** \brief Returns the corresponding centroid (x,y,z) coordinates
    * in the grid of voxel (i,j,k).
    * \param[in] ijk the coordinate (i, j, k) of the voxel
    * \return the (x,y,z) coordinate of the voxel centroid
    */
  inline Eigen::Vector4f
  getCentroidCoordinate (const Eigen::Vector3i& ijk)
  {
    int i,j,k;
    i = ((b_min_[0] < 0) ? (std::abs (min_b_[0]) + ijk[0]) : (ijk[0] - min_b_[0]));
    j = ((b_min_[1] < 0) ? (std::abs (min_b_[1]) + ijk[1]) : (ijk[1] - min_b_[1]));
    k = ((b_min_[2] < 0) ? (std::abs (min_b_[2]) + ijk[2]) : (ijk[2] - min_b_[2]));

    Eigen::Vector4f xyz;
    xyz[0] = b_min_[0] + (leaf_size_[0] * 0.5f) + (static_cast<float> (i) * leaf_size_[0]);
    xyz[1] = b_min_[1] + (leaf_size_[1] * 0.5f) + (static_cast<float> (j) * leaf_size_[1]);
    xyz[2] = b_min_[2] + (leaf_size_[2] * 0.5f) + (static_cast<float> (k) * leaf_size_[2]);
    xyz[3] = 0;
    return xyz;
  }

  inline float
  round (float d)
  {
    return static_cast<float> (std::floor (d + 0.5f));
  }

  // We use round here instead of std::floor due to some numerical issues.
  /** \brief Returns the corresponding (i,j,k) coordinates in the grid of point (x,y,z).
    * \param[in] x the X point coordinate to get the (i, j, k) index at
    * \param[in] y the Y point coordinate to get the (i, j, k) index at
    * \param[in] z the Z point coordinate to get the (i, j, k) index at
    */
  inline Eigen::Vector3i
  getGridCoordinatesRound (float x, float y, float z)
  {
    return Eigen::Vector3i (static_cast<int> (round (x * inverse_leaf_size_[0])),
                            static_cast<int> (round (y * inverse_leaf_size_[1])),
                            static_cast<int> (round (z * inverse_leaf_size_[2])));
  }

  // initialization flag
  bool initialized_;

  Eigen::Vector4f sensor_origin_;
  Eigen::Quaternionf sensor_orientation_;
  std::vector<Eigen::Quaternionf, Eigen::aligned_allocator<Eigen::Quaternionf> > beam_directions_;

  // minimum and maximum bounding box coordinates
  Eigen::Vector4f b_min_, b_max_;

  // voxel grid filtered cloud
  PointCloudFilt filtered_cloud_;

  void initializeVoxelGrid (SubmapObj& submap_i){
      // initialization set to true
      initialized_ = true;
      // create the voxel grid and store the output cloud
      PointCloudT::Ptr cloud_ptr(new PointCloudT);
      *cloud_ptr = submap_i.submap_pcl_;
      this->setInputCloud(cloud_ptr);
      this->filter (filtered_cloud_);
      PointCloudT filtered = filtered_cloud_;

      // Get the minimum and maximum bounding box dimensions
      b_min_[0] = (static_cast<float> ( min_b_[0]) * leaf_size_[0]);
      b_min_[1] = (static_cast<float> ( min_b_[1]) * leaf_size_[1]);
      b_min_[2] = (static_cast<float> ( min_b_[2]) * leaf_size_[2]);
      b_max_[0] = (static_cast<float> ( (max_b_[0]) + 1) * leaf_size_[0]);
      b_max_[1] = (static_cast<float> ( (max_b_[1]) + 1) * leaf_size_[1]);
      b_max_[2] = (static_cast<float> ( (max_b_[2]) + 1) * leaf_size_[2]);

      // set the sensor origin and sensor orientation
      sensor_origin_ << submap_i.submap_tf_.translation(), 0.0;
      sensor_orientation_ = Eigen::Quaternionf(submap_i.submap_tf_.linear());

      Eigen::Matrix3f rot = sensor_orientation_.toRotationMatrix();
      Eigen::Vector3f z_or = rot.col(2).transpose();
      std::cout << "Frame origin " << sensor_origin_.transpose() << std::endl;
      std::cout << "Frame direction " << z_or.transpose() << std::endl;
  }


  void createMBES(double spam, int n_beams){
      float roll_step = spam/(n_beams-1);
      float pitch = 0.0, yaw = 0.0;
      Eigen::Quaternionf q;

      // TODO: n_beams needs to be odd!

      for(int i = -n_beams/2; i<=n_beams/2; i++){
//          std::cout << "step " << i << " and angle " << roll_step*i << std::endl;
          q = Eigen::AngleAxisf(roll_step*i, Eigen::Vector3f::UnitX())
              * Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY())
              * Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ());
          beam_directions_.push_back(Eigen::Quaternionf(sensor_orientation_ * q));
          Eigen::Matrix3f rot_mat = (sensor_orientation_ * q).toRotationMatrix();
          Eigen::Vector3f z_or = rot_mat.col(2).transpose();
//          std::cout << "Beams directions " << z_or.transpose() << std::endl;
      }

  }

  int pingComputation (std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i> >& occluded_voxels,
                     std::vector<int>& idxs){
    if (!initialized_){
        PCL_ERROR ("Voxel grid not initialized; call initializeVoxelGrid () first! \n");
        return -1;
    }

    // reserve space for the ray vector
    int reserve_size = div_b_[0] * div_b_[1] * div_b_[2];
    occluded_voxels.reserve (reserve_size);


    Eigen::Matrix3f rot = sensor_orientation_.toRotationMatrix();
    Eigen::Vector3f z_or = rot.col(2).transpose();
    std::cout << "Sensor direction " << z_or.transpose() << std::endl;
    Eigen::Vector4f direction;
    std::cout << "Minb " << min_b_.transpose() << std::endl;
    std::cout << "Maxb " << max_b_.transpose() << std::endl;

    int cnt = 0;
    // iterate over the entire voxel grid
    for(Eigen::Quaternionf beam_n: beam_directions_){
        for (int kk = min_b_.z (); kk <= max_b_.z (); ++kk){
            for (int jj = min_b_.y (); jj <= max_b_.y (); ++jj){
                for (int ii = min_b_.x (); ii <= max_b_.x (); ++ii){
                    cnt = cnt + 1;
                    Eigen::Vector3i ijk (ii, jj, kk);
                    // process all free voxels
                    int index = this->getCentroidIndexAt (ijk);
                    if (index != -1){
                        // estimate direction to target voxel
                        Eigen::Matrix3f rot_beam = beam_n.toRotationMatrix();
                        direction << rot_beam.col(2), 0.0;
    //                    std::cout << "Beam i direction " << direction.transpose() << std::endl;
                        direction.normalize ();

                        // estimate entry point into the voxel grid
                        float tmin = rayBoxIntersection (sensor_origin_, direction);

                        // ray traversal
                        int state = rayTraversal (ijk, sensor_origin_, direction, tmin);

                        // if voxel is occluded
                        if (state == 0){
//                            std::cout << "tmin " << tmin << std::endl;
                            occluded_voxels.push_back (ijk);
                            idxs.push_back(index);
                        }
                    }
                }
            }
        }
    }
    beam_directions_.clear();
    std::cout << "Voxels " << cnt << std::endl;
    cnt = 0;
    return 0;

  }

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float rayBoxIntersection (const Eigen::Vector4f& origin, const Eigen::Vector4f& direction)
{
float tmin, tmax, tymin, tymax, tzmin, tzmax;

if (direction[0] >= 0)
{
 tmin = (b_min_[0] - origin[0]) / direction[0];
 tmax = (b_max_[0] - origin[0]) / direction[0];
}
else
{
 tmin = (b_max_[0] - origin[0]) / direction[0];
 tmax = (b_min_[0] - origin[0]) / direction[0];
}

if (direction[1] >= 0)
{
 tymin = (b_min_[1] - origin[1]) / direction[1];
 tymax = (b_max_[1] - origin[1]) / direction[1];
}
else
{
 tymin = (b_max_[1] - origin[1]) / direction[1];
 tymax = (b_min_[1] - origin[1]) / direction[1];
}

if ((tmin > tymax) || (tymin > tmax))
{
 PCL_ERROR ("no intersection with the bounding box \n");
 tmin = -1.0f;
 return tmin;
}

if (tymin > tmin)
 tmin = tymin;
if (tymax < tmax)
 tmax = tymax;

if (direction[2] >= 0)
{
 tzmin = (b_min_[2] - origin[2]) / direction[2];
 tzmax = (b_max_[2] - origin[2]) / direction[2];
}
else
{
 tzmin = (b_max_[2] - origin[2]) / direction[2];
 tzmax = (b_min_[2] - origin[2]) / direction[2];
}

if ((tmin > tzmax) || (tzmin > tmax))
{
 PCL_ERROR ("no intersection with the bounding box \n");
 tmin = -1.0f;
 return tmin;
}

if (tzmin > tmin)
 tmin = tzmin;
if (tzmax < tmax)
 tmax = tzmax;

return tmin;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int rayTraversal (const Eigen::Vector3i& target_voxel,
                  const Eigen::Vector4f& origin,
                  const Eigen::Vector4f& direction,
                  const float t_min)
{
// coordinate of the boundary of the voxel grid
Eigen::Vector4f start = origin + t_min * direction;

// i,j,k coordinate of the voxel were the ray enters the voxel grid
Eigen::Vector3i ijk = getGridCoordinatesRound (start[0], start[1], start[2]);

// steps in which direction we have to travel in the voxel grid
int step_x, step_y, step_z;

// centroid coordinate of the entry voxel
Eigen::Vector4f voxel_max = getCentroidCoordinate (ijk);

if (direction[0] >= 0)
{
 voxel_max[0] += leaf_size_[0] * 0.5f;
 step_x = 1;
}
else
{
 voxel_max[0] -= leaf_size_[0] * 0.5f;
 step_x = -1;
}
if (direction[1] >= 0)
{
 voxel_max[1] += leaf_size_[1] * 0.5f;
 step_y = 1;
}
else
{
 voxel_max[1] -= leaf_size_[1] * 0.5f;
 step_y = -1;
}
if (direction[2] >= 0)
{
 voxel_max[2] += leaf_size_[2] * 0.5f;
 step_z = 1;
}
else
{
 voxel_max[2] -= leaf_size_[2] * 0.5f;
 step_z = -1;
}

float t_max_x = t_min + (voxel_max[0] - start[0]) / direction[0];
float t_max_y = t_min + (voxel_max[1] - start[1]) / direction[1];
float t_max_z = t_min + (voxel_max[2] - start[2]) / direction[2];

float t_delta_x = leaf_size_[0] / static_cast<float> (std::abs (direction[0]));
float t_delta_y = leaf_size_[1] / static_cast<float> (std::abs (direction[1]));
float t_delta_z = leaf_size_[2] / static_cast<float> (std::abs (direction[2]));

// index of the point in the point cloud
int index;

while ( (ijk[0] < max_b_[0]+1) && (ijk[0] >= min_b_[0]) &&
       (ijk[1] < max_b_[1]+1) && (ijk[1] >= min_b_[1]) &&
       (ijk[2] < max_b_[2]+1) && (ijk[2] >= min_b_[2]) )
{
 // check if we reached target voxel
 if (ijk[0] == target_voxel[0] && ijk[1] == target_voxel[1] && ijk[2] == target_voxel[2])
   return 0;

 // check if voxel is occupied, if yes return 1 for occluded
 index = this->getCentroidIndexAt (ijk);
 if (index != -1)
   return 1;

 // estimate next voxel
 if(t_max_x <= t_max_y && t_max_x <= t_max_z)
 {
   t_max_x += t_delta_x;
   ijk[0] += step_x;
 }
 else if(t_max_y <= t_max_z && t_max_y <= t_max_x)
 {
   t_max_y += t_delta_y;
   ijk[1] += step_y;
 }
 else
 {
   t_max_z += t_delta_z;
   ijk[2] += step_z;
 }
}
return 0;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int rayTraversal (std::vector<Eigen::Vector3i, Eigen::aligned_allocator<Eigen::Vector3i> >& out_ray,
                  const Eigen::Vector3i& target_voxel,
                  const Eigen::Vector4f& origin,
                  const Eigen::Vector4f& direction,
                  const float t_min)
{
// reserve space for the ray vector
int reserve_size = div_b_.maxCoeff () * div_b_.maxCoeff ();
out_ray.reserve (reserve_size);

// coordinate of the boundary of the voxel grid
Eigen::Vector4f start = origin + t_min * direction;

// i,j,k coordinate of the voxel were the ray enters the voxel grid
Eigen::Vector3i ijk = getGridCoordinatesRound (start[0], start[1], start[2]);
//Eigen::Vector3i ijk = this->getGridCoordinates (start_x, start_y, start_z);

// steps in which direction we have to travel in the voxel grid
int step_x, step_y, step_z;

// centroid coordinate of the entry voxel
Eigen::Vector4f voxel_max = getCentroidCoordinate (ijk);

if (direction[0] >= 0)
{
 voxel_max[0] += leaf_size_[0] * 0.5f;
 step_x = 1;
}
else
{
 voxel_max[0] -= leaf_size_[0] * 0.5f;
 step_x = -1;
}
if (direction[1] >= 0)
{
 voxel_max[1] += leaf_size_[1] * 0.5f;
 step_y = 1;
}
else
{
 voxel_max[1] -= leaf_size_[1] * 0.5f;
 step_y = -1;
}
if (direction[2] >= 0)
{
 voxel_max[2] += leaf_size_[2] * 0.5f;
 step_z = 1;
}
else
{
 voxel_max[2] -= leaf_size_[2] * 0.5f;
 step_z = -1;
}

float t_max_x = t_min + (voxel_max[0] - start[0]) / direction[0];
float t_max_y = t_min + (voxel_max[1] - start[1]) / direction[1];
float t_max_z = t_min + (voxel_max[2] - start[2]) / direction[2];

float t_delta_x = leaf_size_[0] / static_cast<float> (std::abs (direction[0]));
float t_delta_y = leaf_size_[1] / static_cast<float> (std::abs (direction[1]));
float t_delta_z = leaf_size_[2] / static_cast<float> (std::abs (direction[2]));

// the index of the cloud (-1 if empty)
int index = -1;
int result = 0;

while ( (ijk[0] < max_b_[0]+1) && (ijk[0] >= min_b_[0]) &&
       (ijk[1] < max_b_[1]+1) && (ijk[1] >= min_b_[1]) &&
       (ijk[2] < max_b_[2]+1) && (ijk[2] >= min_b_[2]) )
{
 // add voxel to ray
 out_ray.push_back (ijk);

 // check if we reached target voxel
 if (ijk[0] == target_voxel[0] && ijk[1] == target_voxel[1] && ijk[2] == target_voxel[2])
   break;

 // check if voxel is occupied
 index = this->getCentroidIndexAt (ijk);
 if (index != -1)
   result = 1;

 // estimate next voxel
 if(t_max_x <= t_max_y && t_max_x <= t_max_z)
 {
   t_max_x += t_delta_x;
   ijk[0] += step_x;
 }
 else if(t_max_y <= t_max_z && t_max_y <= t_max_x)
 {
   t_max_y += t_delta_y;
   ijk[1] += step_y;
 }
 else
 {
   t_max_z += t_delta_z;
   ijk[2] += step_z;
 }
}
return result;
}
};

#endif // MULTIBEAM_SIMPLE_HPP
