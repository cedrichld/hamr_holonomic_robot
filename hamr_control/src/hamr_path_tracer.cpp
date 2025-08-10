// hamr_path_tracer.cpp
#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include <deque>
#include <chrono>

using namespace std;
using namespace std::chrono_literals;

class HamrPathTracer : public rclcpp::Node {
public:
  HamrPathTracer()
  : Node("hamr_path_tracer"),
    max_points_(declare_parameter("max_points", 2000)),
    publish_rate_hz_(declare_parameter("publish_rate_hz", 10.0))
  {
    path_pub_   = create_publisher<nav_msgs::msg::Path>("/hamr/path", 1);
    marker_pub_ = create_publisher<visualization_msgs::msg::Marker>("/hamr/path_marker", 1);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/hamr/odom", 50, [this](nav_msgs::msg::Odometry::SharedPtr msg){ on_odom(msg); });

    auto period = std::chrono::milliseconds((int) (1000.0 / publish_rate_hz_));
    timer_ = create_wall_timer(period, [this]{ publish_outputs(); });
  }

private:
  void on_odom(const nav_msgs::msg::Odometry::SharedPtr& msg) {
    frame_id_ = msg->header.frame_id.empty() ? string("odom") : msg->header.frame_id;

    geometry_msgs::msg::PoseStamped ps;
    ps.header = msg->header;
    ps.pose   = msg->pose.pose;

    poses_.push_back(ps);
    if ((int)poses_.size() > max_points_) poses_.pop_front();
  }

  void publish_outputs() {
    if (poses_.empty()) return;

    // Path message
    nav_msgs::msg::Path path;
    path.header.stamp = now();
    path.header.frame_id = frame_id_;
    path.poses.assign(poses_.begin(), poses_.end());
    path_pub_->publish(path);

    // Marker (LINE_STRIP)
    visualization_msgs::msg::Marker m;
    m.header = path.header;
    m.ns = "hamr_traj";
    m.id = 0;
    m.type = visualization_msgs::msg::Marker::LINE_STRIP;
    m.action = visualization_msgs::msg::Marker::ADD;
    m.scale.x = 0.02;               // line width (meters)
    m.color.a = 1.0;
    m.color.r = 0.1; m.color.g = 0.8; m.color.b = 0.2;

    m.points.reserve(poses_.size());
    for (auto &ps : poses_) {
      geometry_msgs::msg::Point p;
      p.x = ps.pose.position.x;
      p.y = ps.pose.position.y;
      p.z = ps.pose.position.z;
      m.points.push_back(p);
    }
    marker_pub_->publish(m);
  }

  // pubs/subs
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // state/params
  std::deque<geometry_msgs::msg::PoseStamped> poses_;
  string frame_id_{"odom"};
  int max_points_;
  double publish_rate_hz_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HamrPathTracer>());
  rclcpp::shutdown();
  return 0;
}
