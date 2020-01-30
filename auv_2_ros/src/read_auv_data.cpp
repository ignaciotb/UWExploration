#include <boost/filesystem.hpp>

#include "data_tools/transforms.h"
#include "data_tools/benchmark.h"

#include "data_tools/navi_data.h"
#include "data_tools/std_data.h"
#include "data_tools/gsf_data.h"
#include <data_tools/csv_data.h>
#include <data_tools/xtf_data.h>
#include <data_tools/all_data.h>

#include "submaps_tools/submaps.hpp"
#include "submaps_tools/cxxopts.hpp"

#include "registration/utils_visualization.hpp"

using namespace std;

double angle_limit (double angle) // keep angle within [0;2*pi[
{
    while (angle >= M_PI*2)
        angle -= M_PI*2;
    while (angle < 0)
        angle += M_PI*2;
    return angle;
}

void divide_on_tracks(std_data::mbes_ping::PingsT& pings){

    int num_pings = pings.size();
    int cnt = 0;
    int min_pings = 300;
    int max_pings = 1000;

    double current_heading; // heading of vehicle
    double past_heading = 0; // heading of vehicle
    std::cout << num_pings << std::endl;
    std::cout << "First ping " << pings[0].time_string_ << std::endl;
    std::cout << "Last ping " << pings[num_pings-3].time_string_ << std::endl;
    for(int i=0; i<num_pings; i++){
        pings[i].first_in_file_ = false;
        current_heading = angle_limit(pings[i].heading_);
        if(i==0){
            past_heading = current_heading;
        }
        if ((abs(current_heading- past_heading) > M_PI) && cnt > min_pings /*|| cnt > max_pings*/){
            std::cout << i << std::endl;
            pings[i].first_in_file_ = true;
            past_heading = current_heading;
            cnt = 0;
        }
        cnt++;
    }
}

int main(int argc, char** argv) {
    string folder_str;
    string file_str;
    string type;

    cxxopts::Options options("example_reader", "Reads different mbes and sss file formats and saves them to a common format");
    options.add_options()
      ("help", "Print help")
      ("folder", "Input folder containing mbes files", cxxopts::value(folder_str))
      ("file", "Output file", cxxopts::value(file_str))
      ("type", "Type of data to read, options: all, xtf, navi, gsf", cxxopts::value(type));

    auto result = options.parse(argc, argv);
    if (result.count("help")) {
        cout << options.help({ "", "Group" }) << endl;
        exit(0);
    }

    if (result.count("folder") == 0) {
        cout << "Please provide folder containing mbes or sss files..." << endl;
        exit(0);
    }
    if (result.count("type") == 0) {
        cout << "Please provide input type, options: all, xtf, navi, gsf" << endl;
        exit(0);
    }


    boost::filesystem::path folder(folder_str);
    boost::filesystem::path pings_path = folder / "mbes_pings.cereal";
    boost::filesystem::path submaps_path = folder / "submaps.cereal";

    // if side scan, read and save, then return
    if (type == "xtf") {
        cout << "Not much to do by now" << endl;
        return 0;
    }

    // otherwise we have multibeam data, read and save
    std_data::mbes_ping::PingsT std_pings;
    if (type == "gsf") {
        gsf_data::gsf_mbes_ping::PingsT pings = std_data::parse_folder<gsf_data::gsf_mbes_ping>(folder);
        std::stable_sort(pings.begin(), pings.end(), [](const gsf_data::gsf_mbes_ping& ping1, const gsf_data::gsf_mbes_ping& ping2) {
            return ping1.time_stamp_ < ping2.time_stamp_;
        });
        std_pings = gsf_data::convert_pings(pings);
    }
    else if (type == "all") {
        all_data::all_nav_entry::EntriesT entries = std_data::parse_folder<all_data::all_nav_entry>(folder);
        all_data::all_mbes_ping::PingsT pings = std_data::parse_folder<all_data::all_mbes_ping>(folder);
        std_pings = all_data::convert_matched_entries(pings, entries);
    }
    else if (type == "navi") {
        std_pings = std_data::parse_folder<std_data::mbes_ping>(folder / "Pings");
        std_data::nav_entry::EntriesT entries = std_data::parse_folder<std_data::nav_entry>(folder / "NavUTM");
        navi_data::match_timestamps(std_pings, entries);
    }
    else {
        cout << "Type " << type << " is not supported!" << endl;
        return 0;
    }

    // Read from disk
//    std_pings = std_data::read_data<std_data::mbes_ping::PingsT>(pings_path);

    // Divides the tracks into roughly square pieces
    divide_on_tracks(std_pings);
//    navi_data::divide_tracks_equal(std_pings);

    // Save maps as .xyz pointclouds to use with external tools
//    navi_data::save_submaps_files(std_pings, folder);

    // convert to submaps
    std_data::pt_submaps ss;
    std::tie(ss.points, ss.trans, ss.angles,
             ss.matches,ss.bounds, ss.tracks) = navi_data::create_submaps(std_pings);
    for (const Eigen::Vector3d& ang : ss.angles) {
        ss.rots.push_back(data_transforms::euler_to_matrix(ang(0), ang(1), ang(2)));
    }

    // write to disk
    std_data::write_data(ss, submaps_path);


    return 0;
}
