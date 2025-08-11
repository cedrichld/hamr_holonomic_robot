cmake_minimum_required(VERSION 3.8)
project(hamr_control_cpp)

include_directories(include)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(hamr_interfaces REQUIRED)

# Controller
add_executable(hamr_controller
  src/hamr_controller.cpp
  src/hamr_control_node.cpp
)
ament_target_dependencies(hamr_controller
  rclcpp std_msgs nav_msgs geometry_msgs tf2_msgs hamr_interfaces
)
target_include_directories(hamr_controller PRIVATE include)

target_include_directories(hamr_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Path tracer
add_executable(hamr_path_tracer
  src/hamr_path_tracer.cpp
)
ament_target_dependencies(hamr_path_tracer
  rclcpp nav_msgs visualization_msgs geometry_msgs
)
target_include_directories(hamr_path_tracer PRIVATE include)

install(TARGETS
  hamr_controller
  hamr_path_tracer
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

ament_package()
