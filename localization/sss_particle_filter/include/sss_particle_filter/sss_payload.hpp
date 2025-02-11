#include <iostream>
#include <vector>
#include <cmath>
#include <memory>
#include <chrono>
#include <Eigen/Dense>

#include "data_tools/csv_data.h"
#include "data_tools/xtf_data.h"
#include <bathy_maps/base_draper.h>
#include <bathy_maps/mesh_map.h>

#include "cnpy.h"
#include <igl/embree/EmbreeIntersector.h>

// #include <igl/opengl/glfw/Viewer.h>
// #include <igl/opengl/gl.h>

class DraperWrapper
{

public:
    DraperWrapper(std::string mesh_path);

    template <typename T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> parseMeshComponents(T *data, std::vector<size_t> v_shape)
    {
        size_t v_rows = v_shape[0];
        size_t v_cols = v_shape[1];
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Mat(v_rows, v_cols);

        int row_cnt = 0;
        int col_cnt = 0;
        for (size_t i = 0; i < v_rows; ++i)
        {
            for (size_t j = 0; j < v_cols; ++j)
            {
                Mat(row_cnt, col_cnt) = data[i * v_cols + j];
                row_cnt++;

                if (row_cnt == v_rows)
                {
                    row_cnt = 0;
                    col_cnt++;
                }
            }
        }
        return Mat;
    }

    std::string svp_path;
    double svp;
    double max_r;
    std::unique_ptr<BaseDraper> draper;
    csv_data::csv_asvp_sound_speed::EntriesT isovelocity_sound_speeds;
    xtf_data::xtf_sss_ping xtf_ping_;
};
