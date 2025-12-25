#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>

#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <mutex>
#include <optional>
#include <string>
#include <chrono>
#include <filesystem>

using namespace std::chrono_literals;

class PclMapper : public rclcpp::Node
{
public:
  PclMapper()
  : Node("pcl_mapper"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // Parameters
    input_topic_ = this->declare_parameter<std::string>(
      "input_topic", "/rtabmap/cloud_map");
    target_frame_ = this->declare_parameter<std::string>(
      "target_frame", "map");
    output_dir_ = this->declare_parameter<std::string>(
      "output_dir", "/home/kartik/maps/pcd");
    filename_prefix_ = this->declare_parameter<std::string>(
      "filename_prefix", "hamr_room_");
    save_on_shutdown_ = this->declare_parameter<bool>(
      "save_on_shutdown", true);

    RCLCPP_INFO(get_logger(), "PclMapper subscribing to: %s", input_topic_.c_str());
    RCLCPP_INFO(get_logger(), "Target frame: %s", target_frame_.c_str());
    RCLCPP_INFO(get_logger(), "Output dir: %s", output_dir_.c_str());
    RCLCPP_INFO(get_logger(), "Save on shutdown: %s", save_on_shutdown_ ? "true" : "false");

    // Ensure output dir exists
    std::error_code ec;
    std::filesystem::create_directories(output_dir_, ec);
    if (ec) {
      RCLCPP_WARN(get_logger(), "Could not create output dir %s: %s",
                  output_dir_.c_str(), ec.message().c_str());
    }

    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::QoS(10),
      std::bind(&PclMapper::cloudCallback, this, std::placeholders::_1));
  }

  ~PclMapper() override
  {
    if (save_on_shutdown_) {
      RCLCPP_INFO(get_logger(), "Node shutting down, saving final colored pointcloud...");
      saveFinalCloud();
    }
  }

private:
  void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_cloud_ = *msg;
  }

  void saveFinalCloud()
  {
    sensor_msgs::msg::PointCloud2 cloud;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!latest_cloud_.has_value()) {
        RCLCPP_WARN(get_logger(), "No pointcloud received, nothing to save.");
        return;
      }
      cloud = latest_cloud_.value();
    }

    // Transform to target frame if needed
    sensor_msgs::msg::PointCloud2 cloud_tf;
    if (target_frame_.empty() || cloud.header.frame_id == target_frame_) {
      cloud_tf = cloud;
    } else {
      try {
        auto tf = tf_buffer_.lookupTransform(
          target_frame_, cloud.header.frame_id,
          cloud.header.stamp, 200ms);

        tf2::doTransform(cloud, cloud_tf, tf);
      } catch (const tf2::TransformException & ex) {
        RCLCPP_WARN(get_logger(),
                    "TF unavailable (%s -> %s): %s. Saving in original frame.",
                    cloud.header.frame_id.c_str(),
                    target_frame_.c_str(),
                    ex.what());
        cloud_tf = cloud; // fallback
      }
    }

    // Convert to PCL *with color* (XYZRGB)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
    pcl::fromROSMsg(cloud_tf, *pcl_cloud);

    if (pcl_cloud->empty()) {
      RCLCPP_WARN(get_logger(), "Final cloud is empty, not saving.");
      return;
    }

    // Filename with timestamp
    auto now = this->get_clock()->now().nanoseconds();
    std::string filename = output_dir_ + "/" + filename_prefix_ + std::to_string(now) + ".pcd";

    int ret = pcl::io::savePCDFileBinary(filename, *pcl_cloud);
    if (ret == 0) {
      RCLCPP_INFO(get_logger(), "Saved final colored PCD: %s (%zu points)",
                  filename.c_str(), pcl_cloud->size());
    } else {
      RCLCPP_ERROR(get_logger(), "Failed to save PCD file: %s", filename.c_str());
    }
  }

  // Parameters
  std::string input_topic_;
  std::string target_frame_;
  std::string output_dir_;
  std::string filename_prefix_;
  bool save_on_shutdown_;

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Data
  std::mutex mutex_;
  std::optional<sensor_msgs::msg::PointCloud2> latest_cloud_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PclMapper>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
