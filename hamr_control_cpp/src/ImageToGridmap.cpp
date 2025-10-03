/*
 * ImageToGridmap.cpp
 *
 *  Created on: May 4, 2015
 *      Author: Péter Fankhauser
 *	 Institute: ETH Zurich, ANYbotics
 */

#include <string>
#include <utility>

#include "hamr_control_cpp/ImageToGridmap.hpp"

namespace hamr_control_cpp
{

ImageToGridmap::ImageToGridmap()
: Node("image_to_gridmap_"),
  map_(grid_map::GridMap({"elevation"})),
  mapInitialized_(false)
{
  readParameters();
  map_.setFrameId(mapFrameId_);
  map_.setBasicLayers({"elevation"});
  imageSubscriber_ =
    this->create_subscription<sensor_msgs::msg::Image>(
    imageTopic_, 1,
    std::bind(&ImageToGridmap::imageCallback, this, std::placeholders::_1));

  gridMapPublisher_ = this->create_publisher<grid_map_msgs::msg::GridMap>(
    "grid_map", rclcpp::QoS(1).transient_local());
}

ImageToGridmap::~ImageToGridmap()
{
}

bool ImageToGridmap::readParameters()
{
  this->declare_parameter("image_topic", std::string("/image"));
  this->declare_parameter("resolution", rclcpp::ParameterValue(0.03));
  this->declare_parameter("min_height", rclcpp::ParameterValue(0.0));
  this->declare_parameter("max_height", rclcpp::ParameterValue(1.0));
  this->declare_parameter("map_frame_id", std::string("map"));

  this->get_parameter("image_topic", imageTopic_);
  this->get_parameter("resolution", resolution_);
  this->get_parameter("min_height", minHeight_);
  this->get_parameter("max_height", maxHeight_);
  this->get_parameter("map_frame_id", mapFrameId_);

  RCLCPP_INFO(
    this->get_logger(),
    "Params: image_topic=%s res=%.3f minH=%.3f maxH=%.3f frame=%s",
    imageTopic_.c_str(), resolution_, minHeight_, maxHeight_, mapFrameId_.c_str());

  return true;
}

void ImageToGridmap::imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  if (!mapInitialized_) {
    grid_map::GridMapRosConverter::initializeFromImage(*msg, resolution_, map_);
    // Make sure the frame is correct after geometry init.
    map_.setFrameId(mapFrameId_);

    RCLCPP_INFO(
      this->get_logger(),
      "Initialized map with size %f x %f m (%i x %i cells).", map_.getLength().x(),
      map_.getLength().y(), map_.getSize()(0), map_.getSize()(1));
    mapInitialized_ = true;
  }
  grid_map::GridMapRosConverter::addLayerFromImage(*msg, "elevation", map_, minHeight_, maxHeight_);
  grid_map::GridMapRosConverter::addColorLayerFromImage(*msg, "color", map_);

  // Publish as grid map.
  // (Keep frame consistent even if params might change at runtime.)
  map_.setFrameId(mapFrameId_);
  auto message = grid_map::GridMapRosConverter::toMessage(map_);
  gridMapPublisher_->publish(std::move(message));
}

}  // namespace hamr_control_cpp