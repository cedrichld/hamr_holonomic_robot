#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <deque>
#include <cmath>
#include <string>
#include <tf2/utils.h>

class CompaPathTracer : public rclcpp::Node {
public:
  CompaPathTracer()
  : Node("compa_path_tracer"),
    max_points_(declare_parameter("max_points", 20000)),
    publish_rate_hz_(declare_parameter("publish_rate_hz", 10.0)),
    offset_x_(declare_parameter("offset_x", 0.0)),  // forward in robot frame
    offset_y_(declare_parameter("offset_y", 0.0)),  // left in robot frame
    offset_z_(declare_parameter("offset_z", 0.0))   // up in robot frame
  {
    path_pub_   = create_publisher<nav_msgs::msg::Path>("/compa/path", 1);
    marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("/compa/path_marker", 1);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/compa/odom", 50,
      [this](nav_msgs::msg::Odometry::SharedPtr msg){ on_odom(msg); });

    auto period = std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate_hz_));
    timer_ = create_wall_timer(period, [this]{ publish_outputs(); });

    RCLCPP_INFO(get_logger(),
      "CompaPathTracer started with max_points=%d, rate=%.1f Hz, offset=(%.3f, %.3f, %.3f)",
      max_points_, publish_rate_hz_, offset_x_, offset_y_, offset_z_);
  }

private:
  void on_odom(const nav_msgs::msg::Odometry::SharedPtr& msg) {
    frame_id_ = msg->header.frame_id.empty() ? "odom" : msg->header.frame_id;

    // Extract yaw
    double yaw = tf2::getYaw(msg->pose.pose.orientation);

    // Rotate offset from body frame to odom frame
    double dx = std::cos(yaw) * offset_x_ - std::sin(yaw) * offset_y_;
    double dy = std::sin(yaw) * offset_x_ + std::cos(yaw) * offset_y_;

    geometry_msgs::msg::PoseStamped ps;
    ps.header = msg->header;
    ps.pose   = msg->pose.pose;

    // Apply offset to get center position
    ps.pose.position.x += dx;
    ps.pose.position.y += dy;
    ps.pose.position.z += offset_z_;

    poses_.push_back(ps);
    if ((int)poses_.size() > max_points_) poses_.pop_front();
  }

  void publish_outputs() {
    if (poses_.empty()) return;

    nav_msgs::msg::Path path;
    path.header.stamp = poses_.back().header.stamp;
    path.header.frame_id = frame_id_;
    path.poses.assign(poses_.begin(), poses_.end());
    path_pub_->publish(path);

    visualization_msgs::msg::Marker m;
    m.header = path.header;
    m.ns = "compa_traj";
    m.id = 0;
    m.type = visualization_msgs::msg::Marker::LINE_STRIP;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = 0.02;
    m.color.a = 1.0;
    m.color.r = 0.1; m.color.g = 0.8; m.color.b = 0.2;

    m.points.reserve(poses_.size());
    for (auto &ps : poses_) {
      m.points.push_back(ps.pose.position);
    }
    marker_pub_->publish(m);
  }

  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  std::deque<geometry_msgs::msg::PoseStamped> poses_;
  std::string frame_id_;
  int max_points_;
  double publish_rate_hz_;
  double offset_x_, offset_y_, offset_z_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CompaPathTracer>());
  rclcpp::shutdown();
  return 0;
}
