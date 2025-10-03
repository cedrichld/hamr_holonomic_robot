/*
 * image_to_gridmap_demo_node.cpp
 *
 *  Created on: May 04, 2015
 *      Author: Martin Wermelinger
 *   Institute: ETH Zurich, ANYbotics
 */

#include <rclcpp/rclcpp.hpp>
#include <memory>
#include "hamr_control_cpp/ImageToGridmap.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<hamr_control_cpp::ImageToGridmap>());
  rclcpp::shutdown();
  return 0;
}