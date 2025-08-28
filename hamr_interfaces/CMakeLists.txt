cmake_minimum_required(VERSION 3.8)
project(hamr_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Generate message files
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/StateError.msg"
  "msg/LiveGains.msg"
  "msg/ReferenceTraj.msg"
)

# Add any additional runtime dependencies
ament_export_dependencies(rosidl_default_runtime)

# Install the package
ament_package()
