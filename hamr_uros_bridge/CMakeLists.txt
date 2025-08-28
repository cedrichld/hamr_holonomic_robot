cmake_minimum_required(VERSION 3.8)
project(hamr_uros_bridge)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)

add_executable(relay_node src/relay_node.cpp)
ament_target_dependencies(relay_node rclcpp std_msgs)

install(TARGETS relay_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()
