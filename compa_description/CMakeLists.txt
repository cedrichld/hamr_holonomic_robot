cmake_minimum_required(VERSION 3.8)
project(compa_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(xacro REQUIRED)

install(
  DIRECTORY urdf meshes
  DESTINATION share/${PROJECT_NAME}/
)

# xacro_add_files(
#   urdf/compa.sdf.xacro
#   INSTALL DESTINATION share/${PROJECT_NAME}/urdf
# )

ament_package()
