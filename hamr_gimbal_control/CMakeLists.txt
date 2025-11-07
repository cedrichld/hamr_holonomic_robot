cmake_minimum_required(VERSION 3.8)
project(hamr_gimbal_control)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(gimbal_control_node src/gimbal_control_node.cpp)
ament_target_dependencies(gimbal_control_node rclcpp std_msgs)

install(TARGETS gimbal_control_node DESTINATION lib/${PROJECT_NAME})

ament_package()
