cmake_minimum_required(VERSION 3.12)

# Set CMP0048 to NEW to allow project VERSION specification
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif()

project(auv_model VERSION 1.0.0)
include(ExternalProject)


# # Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
    roscpp
    eigen_conversions
    tf
    actionlib_msgs
    actionlib
    std_msgs
    sensor_msgs
    #auv_2_ros
)

# AUVLIB stuff
find_package(LibXTF REQUIRED)
find_package(LibGSF REQUIRED)
find_package(EigenCereal REQUIRED)
find_package(DataTools REQUIRED)
find_package(SonarTracing REQUIRED)
find_package(AuvlibGlad REQUIRED)
find_package(BathyMaps REQUIRED)

find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

# find_library(xtf_data_LIBRARY NAMES libxtf_data PATHS /usr/local/lib/)

set_target_properties(xtf_data PROPERTIES 
    IMPORTED_LOCATION "/usr/local/lib/libxtf_data.a"
)

set_target_properties(std_data PROPERTIES 
    IMPORTED_LOCATION "/usr/local/lib/libstd_data.a"
)

set_target_properties(mesh_map PROPERTIES 
    IMPORTED_LOCATION "/usr/local/lib/libmesh_map.a"
)

set_target_properties(base_draper PROPERTIES 
    IMPORTED_LOCATION "/usr/local/lib/libbase_draper.a"
)


# if(xtf_data_LIBRARY)
#     target_link_libraries(libsss_payload ${xtf_data_LIBRARY})
# else()
#     message(WARNING "------------------------xtf_data not found; libsss_payload might be missing functionality.")
# endif()

#if (LIBXTF_FOUND):
# message("Message=================================: " ${LIBXTF_FOUND})
	#endif()

# Libigl for raytracing
set(EMBREE_ISPC_SUPPORT OFF)
set(EMBREE_TUTORIALS OFF)
# Install this https://github.com/libigl/libigl-example-project/tree/main
list(PREPEND CMAKE_MODULE_PATH "/home/sam/auv_ws/src/libigl-example-project/cmake")
# Add as a set of headers
include(libigl) 
# Add modules here
igl_include(glfw)
igl_include(embree)

include_directories(${LIBIGL_INCLUDE_DIR})
include_directories(${EMBREE_INCLUDE_DIRS})
include_directories(/home/sam/workspaces/auvlib/src/sonar_tracing/include)

# list(PREPEND CMAKE_MODULE_PATH "/home/sam/auv_ws/src/libigl-example-project/cmake")


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

# Generate messages in the 'msg' folder
add_message_files(
  FILES
  Sidescan.msg
)

# Generate actions in the 'action' folder
 add_action_files(
   FILES
   MbesSim.action
   SssSim.action
 )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
   actionlib_msgs
   geometry_msgs
   sensor_msgs
 )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES sss_payload
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS xtf_data
)

###########
## Build ##
###########

# To include lib cnpy
add_subdirectory(/home/sam/auv_ws/src/cnpy/ [/home/sam/auv_ws/src/cnpy/])

# add_subdirectory(/home/sam/workspaces/auvlib/ [/home/sam/workspaces/auvlib/])


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
 include
 ${catkin_INCLUDE_DIRS}
 ${OpenCV_INCLUDE_DIRS}
 ${EIGEN3_INCLUDE_DIRS}
 /home/sam/auv_ws/src/cnpy/
#  /home/sam/workspaces/auvlib/
#  /home/sam/auv_ws/src/libigl-example-project/
)

# Declare a C++ library
add_library(sss_payload
  src/sss_payload.cpp
)

target_include_directories(sss_payload PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    PRIVATE src)


add_executable(auv_motion_model
       src/auv_motion_simple_node.cpp
       src/auv_motion_simple.cpp
)

add_executable(auv_payload_cpp
  src/auv_payload.cpp
)

# add_executable(view_sidescan_particles
#   src/view_sidescan_particles.cpp
# )

# add_executable(test_ray
#   src/test_ray.cpp
# )

## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(auv_motion_model
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS})

add_dependencies(auv_payload_cpp
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS})

# add_dependencies(view_sidescan_particles
#   ${${PROJECT_NAME}_EXPORTED_TARGETS}
#   ${catkin_EXPORTED_TARGETS})

# add_dependencies(test_ray
#   ${${PROJECT_NAME}_EXPORTED_TARGETS})

# Specify libraries to link a library or executable target against
target_link_libraries(sss_payload
    igl::embree 
    cnpy
    z    
    xtf_data 
    std_data
    mesh_map 
    base_draper 
    bathy_tracer
)

target_link_libraries(auv_motion_model
    ${catkin_LIBRARIES}
)

# # target_link_libraries(view_sidescan_particles
# #     ${catkin_LIBRARIES}
# # )

target_link_libraries(auv_payload_cpp
    ${OpenCV_LIBRARIES}
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES}
    # igl::embree 
    sss_payload
)


#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# install(TARGETS sss_payload EXPORT SssPayloadConfig
#     ARCHIVE  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#     LIBRARY  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#     RUNTIME  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
# set(CMAKE_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}")
# install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# ## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.hpp"
#   PATTERN ".svn" EXCLUDE
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS auv_payload_cpp
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_auv_model.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
