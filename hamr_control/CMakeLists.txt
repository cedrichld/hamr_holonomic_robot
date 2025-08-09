cmake_minimum_required(VERSION 3.8)
project(hamr_control)

# Find ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(hamr_interfaces REQUIRED)


# Add the source files for compilation
add_executable(hamr_controller src/hamr_controller.cpp src/hamr_control_node.cpp)

# Link necessary dependencies
ament_target_dependencies(hamr_controller
  rclcpp
  std_msgs
  nav_msgs
  geometry_msgs
  tf2_msgs
  hamr_interfaces
)

# Add include directory for headers
target_include_directories(hamr_controller
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include  # Include directory where headers are located
  /opt/ros/jazzy/include  # Include ROS 2 dependencies
)

# Install the executable
install(TARGETS hamr_controller
  RUNTIME DESTINATION lib/${PROJECT_NAME})

ament_package()
