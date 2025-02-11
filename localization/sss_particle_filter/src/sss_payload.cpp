#include "sss_particle_filter/sss_payload.hpp"

DraperWrapper::DraperWrapper(std::string mesh_path)
{
    // Read vertices
    cnpy::NpyArray npyVertices = cnpy::npy_load(mesh_path + "/vertices.npy");
    double *vertices = npyVertices.data<double>();
    std::vector<size_t> v_shape = npyVertices.shape;
    Eigen::MatrixXd V = parseMeshComponents<double>(vertices, v_shape);

    // Read edges
    cnpy::NpyArray npyEdges = cnpy::npy_load(mesh_path + "/edges.npy");
    int *edges = npyEdges.data<int>();
    std::vector<size_t> e_shape = npyEdges.shape;
    Eigen::MatrixXi F = parseMeshComponents<int>(edges, e_shape);

    // Read bounds
    cnpy::NpyArray npyBounds = cnpy::npy_load(mesh_path + "/bounds.npy");
    double *bounds = npyBounds.data<double>();
    std::vector<size_t> b_shape = npyBounds.shape;
    mesh_map::BoundsT B = parseMeshComponents<double>(bounds, b_shape);

    svp = 1431.1 - 1; // Mean sound speed (m/s)
    csv_data::csv_asvp_sound_speed iso_sound_speed;
    iso_sound_speed.dbars = Eigen::VectorXd::LinSpaced(50, 0, 49);
    iso_sound_speed.vels = Eigen::VectorXd::Constant(50, svp);
    isovelocity_sound_speeds.push_back(iso_sound_speed);

    max_r = 40.0 / 1500.0 * svp; // Adjust max range for sound speed

    // Build draper
    draper = std::make_unique<BaseDraper>(V, F, B, isovelocity_sound_speeds);
    draper->set_sidescan_yaw(0);
    Eigen::Vector3d sensor_offset(0.0, 0.0, 0.0);
    draper->set_sidescan_port_stbd_offsets(sensor_offset, sensor_offset);

    std::cout << "Draper created" << std::endl;
}