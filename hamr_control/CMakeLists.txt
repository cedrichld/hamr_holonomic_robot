cmake_minimum_required(VERSION 3.8)
project(hamr_control)

# Dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(hamr_interfaces REQUIRED)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install Python node executable
install(PROGRAMS
  hamr_control/hamr_odom_graph.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME hamr_odom_graph
)

# C++ executable
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
  hamr_interfaces
)
target_include_directories(hamr_controller
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Install C++ executable
install(TARGETS hamr_controller
  DESTINATION lib/${PROJECT_NAME}
)


install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)


ament_package()
