/*
 * ImageToGridmap.hpp
 *
 *  Created on: May 4, 2015
 *      Author: Martin Wermelinger
 *	 Institute: ETH Zurich, ANYbotics
 *
 */

#ifndef HAMR_CONTROL_CPP__IMAGE_TO_GRIDMAP_HPP_
#define HAMR_CONTROL_CPP__IMAGE_TO_GRIDMAP_HPP_

// ROS
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <grid_map_ros/grid_map_ros.hpp>

#include <string>

namespace hamr_control_cpp
{

/*!
 * Loads an image and saves it as layer 'elevation' of a grid map.
 * The grid map is published and can be viewed in Rviz.
 */
class ImageToGridmap : public rclcpp::Node
{
public:
  /*!
   * Constructor.
   * @param nodeHandle the ROS node handle.
   */
  ImageToGridmap();

  /*!
   * Destructor.
   */
  virtual ~ImageToGridmap();

  /*!
  * Reads and verifies the ROS parameters.
  * @return true if successful.
  */
  bool readParameters();

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);

private:
  //! Grid map publisher.
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr gridMapPublisher_;

  //! Grid map data.
  grid_map::GridMap map_;

  //! Image subscriber
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr imageSubscriber_;

  //! Name of the grid map topic.
  std::string imageTopic_;

  //! Resolution of the grid map.
  double resolution_;
  double map_width_;
  int img_px_res_;

  //! Range of the height values.
  double minHeight_;
  double maxHeight_;

  //! Frame id of the grid map.
  std::string mapFrameId_;

  bool mapInitialized_;
};

}  // namespace hamr_control_cpp
#endif  // HAMR_CONTROL_CPP__IMAGE_TO_GRIDMAP_HPP_