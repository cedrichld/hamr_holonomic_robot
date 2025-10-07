cmake_minimum_required(VERSION 3.8)
project(hamr_uros_bridge)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(Eigen3 REQUIRED)

add_executable(relay_node src/relay_node.cpp)
ament_target_dependencies(relay_node 
rclcpp 
std_msgs
geometry_msgs
nav_msgs
)
install(TARGETS relay_node
  DESTINATION lib/${PROJECT_NAME})

add_executable(odometry_eval src/odometry_eval.cpp)
ament_target_dependencies(odometry_eval 
rclcpp 
geometry_msgs 
nav_msgs 
message_filters)
target_include_directories(odometry_eval PRIVATE ${EIGEN3_INCLUDE_DIRS})
install(TARGETS odometry_eval 
  DESTINATION lib/${PROJECT_NAME})

ament_package()
