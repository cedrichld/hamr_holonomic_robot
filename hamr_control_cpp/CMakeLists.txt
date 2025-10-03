cmake_minimum_required(VERSION 3.8)
project(hamr_control_cpp)

include_directories(include)
find_package(Eigen3 REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(hamr_interfaces REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_path(NANOFANN_INCLUDE_DIR nanoflann.hpp)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs imgproc)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED) 
find_package(grid_map_core REQUIRED)
find_package(grid_map_ros REQUIRED)
find_package(grid_map_msgs REQUIRED)

# Controller
add_executable(hamr_controller
  src/hamr_controller.cpp
  src/hamr_control_node.cpp
)
ament_target_dependencies(hamr_controller
  rclcpp 
  std_msgs 
  nav_msgs 
  geometry_msgs 
  tf2_msgs 
  tf2
  tf2_ros
  tf2_geometry_msgs
  hamr_interfaces
  grid_map_ros
)
target_include_directories(hamr_controller PRIVATE include)
target_link_libraries(hamr_controller Eigen3::Eigen)

target_include_directories(hamr_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Path tracer
add_executable(hamr_path_tracer
  src/hamr_path_tracer.cpp
)
ament_target_dependencies(hamr_path_tracer
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_geometry_msgs
)
target_include_directories(hamr_path_tracer PRIVATE include)

# Map Publisher
add_executable(map_publisher
  src/map_publisher.cpp
)
ament_target_dependencies(map_publisher
  rclcpp
  nav_msgs
  ament_index_cpp
)
target_include_directories(map_publisher PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(map_publisher yaml-cpp ${OpenCV_LIBS})

# PNG Costmap Publisher
add_executable(cost_map_publisher
  src/cost_map_publisher.cpp
)
ament_target_dependencies(cost_map_publisher
  rclcpp
  nav_msgs
  geometry_msgs
  sensor_msgs
  cv_bridge 
  grid_map_core
  grid_map_ros
  grid_map_msgs
)
target_include_directories(cost_map_publisher PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cost_map_publisher ${OpenCV_LIBS})

# PRM Builder
add_executable(prm_builder
  src/prm_builder.cpp
)
ament_target_dependencies(prm_builder
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
)
if(NANOFANN_INCLUDE_DIR)
  target_include_directories(prm_builder PRIVATE ${NANOFANN_INCLUDE_DIR})
endif()

# ASTAR
add_executable(astar_search
  src/astar_search.cpp
)
ament_target_dependencies(astar_search
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

add_executable(compa_path_tracer
  src/compa_path_tracer.cpp
)

ament_target_dependencies(compa_path_tracer
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_geometry_msgs
)

# Off-ASTAR
add_executable(or_planner
  src/or_planner.cpp
)
ament_target_dependencies(or_planner
  rclcpp
  nav_msgs
  geometry_msgs
  visualization_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  grid_map_core
  grid_map_ros
  grid_map_msgs
)

add_executable(
  image_to_gridmap
  src/image_to_gridmap_node.cpp
  src/ImageToGridmap.cpp
)
ament_target_dependencies(image_to_gridmap
  grid_map_msgs
  grid_map_ros
  rclcpp
)

install(TARGETS
  hamr_controller
  hamr_path_tracer
  compa_path_tracer
  map_publisher
  cost_map_publisher
  prm_builder
  astar_search
  or_planner
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(
  TARGETS image_to_gridmap
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()
