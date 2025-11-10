#include <memory>
#include <mutex>
#include <string>
#include <chrono>
#include <filesystem>  // C++17 for directory creation

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_srvs/srv/trigger.hpp>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/memory.h>

using namespace std::chrono_literals;
using PointT = pcl::PointXYZRGB;
namespace fs = std::filesystem;

class PclMapper : public rclcpp::Node {
public:
  PclMapper() : Node("pcl_mapper"),
                tf_buffer_(this->get_clock()),
                tf_listener_(tf_buffer_) {

    // Params
    target_frame_   = this->declare_parameter<std::string>("target_frame", "map");
    input_topic_    = this->declare_parameter<std::string>("input_topic", "/camera/depth/color/points");
    leaf_size_      = this->declare_parameter<double>("leaf_size", 0.03); // 3 cm
    save_interval_s_= this->declare_parameter<int>("save_interval_s", 0); // 0 = off
    output_pcd_path_= this->declare_parameter<std::string>("output_pcd_path", "map.pcd");
    publish_map_    = this->declare_parameter<bool>("publish_map", true);

    // Create output directory if it doesn't exist
    createOutputDirectory();

    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&PclMapper::cloudCb, this, std::placeholders::_1));

    if (publish_map_) {
      pub_map_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("pcl_map", 1);
      pub_timer_ = this->create_wall_timer(500ms, [this] { publishMap(); });
    }

    save_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "save_pcd",
      [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
             std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        std::lock_guard<std::mutex> lock(mutex_);
        res->success = savePCDUnlocked();
        res->message = res->success ? "Saved PCD" : "Save failed";
      });

    if (save_interval_s_ > 0) {
      save_timer_ = this->create_wall_timer(
        std::chrono::seconds(save_interval_s_),
        [this]{ std::lock_guard<std::mutex> lock(mutex_); savePCDUnlocked(); });
    }

    RCLCPP_INFO(get_logger(), "PCL mapper started. Subscribing: %s, target_frame: %s",
                input_topic_.c_str(), target_frame_.c_str());
    RCLCPP_INFO(get_logger(), "Output: %s, leaf_size: %.3f m, auto_save: %d s",
                output_pcd_path_.c_str(), leaf_size_, save_interval_s_);
  }

private:
  void createOutputDirectory() {
    try {
      fs::path output_path(output_pcd_path_);
      fs::path parent_dir = output_path.parent_path();
      
      if (!parent_dir.empty() && !fs::exists(parent_dir)) {
        fs::create_directories(parent_dir);
        RCLCPP_INFO(get_logger(), "Created output directory: %s", parent_dir.c_str());
      }
    } catch (const fs::filesystem_error& e) {
      RCLCPP_ERROR(get_logger(), "Failed to create output directory: %s", e.what());
    }
  }

  void cloudCb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // Transform to target frame
    sensor_msgs::msg::PointCloud2 cloud_tf;
    try {
      geometry_msgs::msg::TransformStamped tf =
        tf_buffer_.lookupTransform(target_frame_, msg->header.frame_id, msg->header.stamp, 100ms);
      tf2::doTransform(*msg, cloud_tf, tf);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                           "TF unavailable: %s", ex.what());
      return;
    }

    // Convert to PCL
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    pcl::fromROSMsg(cloud_tf, *cloud);

    if (cloud->empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Received empty point cloud");
      return;
    }

    // Downsample
    pcl::PointCloud<PointT>::Ptr cloud_ds(new pcl::PointCloud<PointT>());
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(cloud);
    vg.setLeafSize(static_cast<float>(leaf_size_), static_cast<float>(leaf_size_), static_cast<float>(leaf_size_));
    vg.filter(*cloud_ds);

    // Accumulate
    std::lock_guard<std::mutex> lock(mutex_);
    *global_map_ += *cloud_ds;
    dirty_ = true;
    
    // Log progress occasionally
    static size_t last_logged_size = 0;
    if (global_map_->size() - last_logged_size > 10000) {
      RCLCPP_INFO(get_logger(), "Map now contains %zu points", global_map_->size());
      last_logged_size = global_map_->size();
    }
  }

  void publishMap() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!dirty_ || global_map_->empty()) return;
    
    sensor_msgs::msg::PointCloud2 msg;
    pcl::toROSMsg(*global_map_, msg);
    msg.header.stamp = now();
    msg.header.frame_id = target_frame_;
    pub_map_->publish(msg);
    dirty_ = false;
  }

  bool savePCDUnlocked() {
    if (global_map_->empty()) {
      RCLCPP_WARN(get_logger(), "Map empty—nothing to save.");
      return false;
    }
    
    // Optional final downsample before save
    pcl::PointCloud<PointT>::Ptr map_final(new pcl::PointCloud<PointT>());
    pcl::VoxelGrid<PointT> vg;
    vg.setInputCloud(global_map_);
    vg.setLeafSize(static_cast<float>(leaf_size_), static_cast<float>(leaf_size_), static_cast<float>(leaf_size_));
    vg.filter(*map_final);

    // Ensure directory exists (in case it was deleted during runtime)
    createOutputDirectory();

    int ret = pcl::io::savePCDFileBinary(output_pcd_path_, *map_final);
    if (ret == 0) {
      RCLCPP_INFO(get_logger(), "✓ Saved PCD: %s (%zu points, %.2f MB)",
                  output_pcd_path_.c_str(), map_final->size(),
                  fs::file_size(output_pcd_path_) / (1024.0 * 1024.0));
      return true;
    } else {
      RCLCPP_ERROR(get_logger(), "PCD save failed (%d) -> %s", ret, output_pcd_path_.c_str());
      return false;
    }
  }

  // Params
  std::string target_frame_, input_topic_, output_pcd_path_;
  double leaf_size_;
  int save_interval_s_;
  bool publish_map_;

  // ROS
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_;
  rclcpp::TimerBase::SharedPtr save_timer_, pub_timer_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_srv_;

  // TF
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Map
  pcl::PointCloud<PointT>::Ptr global_map_ = pcl::make_shared<pcl::PointCloud<PointT>>();
  bool dirty_ = false;
  std::mutex mutex_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PclMapper>());
  rclcpp::shutdown();
  return 0;
}