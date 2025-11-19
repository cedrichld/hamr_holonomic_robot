cmake_minimum_required(VERSION 3.8)
project(hamr_pcl_generator)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)              
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_executable(pcl_mapper src/pcl_mapper.cpp)
ament_target_dependencies(pcl_mapper
  rclcpp
  sensor_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  pcl_conversions
  PCL
)

target_include_directories(pcl_mapper PUBLIC
  ${PCL_INCLUDE_DIRS}
)

# Some PCL builds need these; harmless if not required
add_definitions(${PCL_DEFINITIONS})

target_link_libraries(pcl_mapper
  ${PCL_LIBRARIES}
)

install(
  TARGETS pcl_mapper
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config          
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
