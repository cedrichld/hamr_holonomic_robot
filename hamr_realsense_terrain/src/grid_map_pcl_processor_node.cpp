#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <grid_map_ros/grid_map_ros.hpp>
#include <grid_map_msgs/msg/grid_map.hpp>
#include <pcl_conversions/pcl_conversions.h>

#include "grid_map_pcl/GridMapPclLoader.hpp"
#include "grid_map_pcl/helpers.hpp"

namespace gm = ::grid_map::grid_map_pcl;

class GridMapPclProcessorNode : public rclcpp::Node {
public:
  GridMapPclProcessorNode()
  : Node("grid_map_pcl_processor_node")
  {
    // params
    declare_parameter<std::string>("points_topic", "/camera/camera/depth/color/points");
    declare_parameter<std::string>("map_topic", "/elevation_map");
    declare_parameter<std::string>("params_file", "");
    declare_parameter<std::string>("map_frame", "map"); // or base_link if you prefer local
    get_parameter("points_topic", points_topic_);
    get_parameter("map_topic", map_topic_);
    get_parameter("params_file", params_file_);
    get_parameter("map_frame", map_frame_);

    // verbose flag handled by helper
    // gm::setVerbosityLevelToDebugIfFlagSet(shared_from_this());

    pub_ = create_publisher<grid_map_msgs::msg::GridMap>(map_topic_, rclcpp::SystemDefaultsQoS());

    sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      points_topic_,
      rclcpp::SensorDataQoS(),
      std::bind(&GridMapPclProcessorNode::cloudCb, this, std::placeholders::_1));

    // set up loader (works for raw clouds too)
    loader_ = std::make_unique<grid_map::GridMapPclLoader>(this->get_logger());

    if (!params_file_.empty()) {
      loader_->loadParameters(params_file_); // YAML from config
    }

    RCLCPP_INFO(get_logger(), "Listening to: %s, publishing GridMap: %s",
                points_topic_.c_str(), map_topic_.c_str());
  }

private:
  void cloudCb(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    // 1) ROS2 → PCL
    using Point = pcl::PointXYZ;
    pcl::PointCloud<Point>::Ptr cloud(new pcl::PointCloud<Point>());
    pcl::fromROSMsg(*msg, *cloud);

    // (Optional) transform cloud to target frame before processing if you want a world-fixed map.
    // You can also use gm::transformCloud(...) from helpers.hpp given a rigid transform.
    // For TF-based transforms, integrate tf2 before this step.

    // 2) Live processing through GridMapPclLoader API (supports raw clouds)
    loader_->setInputCloud(cloud);                                  // feed input cloud
    loader_->preProcessInputCloud();                                 // filters/outlier removal/downsample
    loader_->initializeGridMapGeometryFromInputCloud();              // set grid extents/resolution
    loader_->addLayerFromInputCloud("elevation");                    // compute elevation layer

    // 3) Publish GridMap
    grid_map::GridMap gridMap = loader_->getGridMap();
    gridMap.setFrameId(map_frame_);
    // grid_map_msgs::msg::GridMap out;
    // grid_map::GridMapRosConverter::toMessage(gridMap, out);
    // pub_->publish(out);
    auto msg_ptr = grid_map::GridMapRosConverter::toMessage(gridMap);           // or toMessage(gridMap, {"elevation"})
    pub_->publish(std::move(*msg_ptr));
  }

  std::string points_topic_, map_topic_, params_file_, map_frame_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<grid_map_msgs::msg::GridMap>::SharedPtr pub_;
  std::unique_ptr<grid_map::GridMapPclLoader> loader_;
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GridMapPclProcessorNode>());
  rclcpp::shutdown();
  return 0;
}
