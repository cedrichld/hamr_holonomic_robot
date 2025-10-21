cmake_minimum_required(VERSION 3.8)
project(hamr_realsense_terrain)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcutils REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(grid_map_core REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(grid_map_msgs REQUIRED)
find_package(grid_map_pcl REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)

include_directories(
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
)

add_executable(grid_map_pcl_processor_node
  src/grid_map_pcl_processor_node.cpp
)

ament_target_dependencies(grid_map_pcl_processor_node
  rclcpp rcutils sensor_msgs
  grid_map_core grid_map_ros grid_map_msgs grid_map_pcl pcl_conversions
)

target_link_libraries(grid_map_pcl_processor_node
  ${PCL_LIBRARIES}
)

install(TARGETS grid_map_pcl_processor_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
