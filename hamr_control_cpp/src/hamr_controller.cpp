#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <std_msgs/msg/float64.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <cmath>
#include <memory>

class HamrPointToPoint : public rclcpp::Node {
public:
  HamrPointToPoint()
  : rclcpp::Node("hamr_point_to_point"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_) {

    // pubs
    left_pub_  = create_publisher<std_msgs::msg::Float64>("/left_wheel/cmd_vel", 10);
    right_pub_ = create_publisher<std_msgs::msg::Float64>("/right_wheel/cmd_vel", 10);
    turret_pub_= create_publisher<std_msgs::msg::Float64>("/turret/cmd_vel", 10);

    // subs
    path_sub_ = create_subscription<nav_msgs::msg::Path>(
      "/astar/path", 10, std::bind(&HamrPointToPoint::onPath, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/hamr/odom", 20, std::bind(&HamrPointToPoint::onOdom, this, std::placeholders::_1));

    // params
    wheel_base_         = declare_parameter("wheel_base_m", 0.19682);
    linear_speed_       = declare_parameter("linear_speed_mps", 0.30);     // fixed v
    angular_speed_      = declare_parameter("angular_speed_rps", 0.50);    // fixed w
    position_tolerance_ = declare_parameter("position_tolerance_m", 0.08); // “at WP”
    align_tolerance_    = declare_parameter("align_tolerance_deg", 8.0) * M_PI/180.0; // face WP
    settle_time_s_      = declare_parameter("settle_time_s", 0.25);
    start_wp_idx_       = declare_parameter("start_waypoint_index", 1);    // 0-based; 1 == “waypoint 2”

    timer_ = create_wall_timer(std::chrono::milliseconds(20), std::bind(&HamrPointToPoint::tick, this));

    RCLCPP_INFO(get_logger(), "PTP follower: v=%.2f m/s, w=%.2f rad/s, start_wp=%zu",
                linear_speed_, angular_speed_, start_wp_idx_);
  }

private:
  enum class Phase { IDLE, ALIGN, DRIVE, SETTLE, DONE };

  // I/O
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_pub_, right_pub_, turret_pub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // TF
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // params
  double wheel_base_{};
  double linear_speed_{};
  double angular_speed_{};
  double position_tolerance_{};
  double align_tolerance_{};
  double settle_time_s_{};
  size_t start_wp_idx_{};

  // state
  nav_msgs::msg::Path::SharedPtr path_;
  nav_msgs::msg::Odometry::SharedPtr odom_;
  size_t wp_idx_{0};
  Phase phase_{Phase::IDLE};
  rclcpp::Time settle_start_;

  // utils
  static double yawOf(const geometry_msgs::msg::Quaternion& q) {
    return std::atan2(2.0*(q.w*q.z + q.x*q.y), 1.0 - 2.0*(q.y*q.y + q.z*q.z));
  }
  static double angNorm(double a){
    while (a >  M_PI) a -= 2*M_PI;
    while (a <= -M_PI) a += 2*M_PI;
    return a;
  }
  void wheels(double v, double w){
    std_msgs::msg::Float64 L,R,T;
    L.data = v - (w*wheel_base_)/2.0;
    R.data = v + (w*wheel_base_)/2.0;
    T.data = 0.0;
    left_pub_->publish(L); right_pub_->publish(R); turret_pub_->publish(T);
  }
  void stop(){ wheels(0.0, 0.0); }

  bool mapToOdom(const geometry_msgs::msg::PoseStamped& in_map, geometry_msgs::msg::PoseStamped& out_odom){
    try {
      auto tf = tf_buffer_.lookupTransform("odom", in_map.header.frame_id, tf2::TimePointZero);
      tf2::doTransform(in_map, out_odom, tf);
      return true;
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "Missing TF %s->odom: %s",
                           in_map.header.frame_id.c_str(), ex.what());
      return false;
    }
  }

  // callbacks
  void onPath(const nav_msgs::msg::Path::SharedPtr msg){
    if (!msg || msg->poses.empty()) {
      RCLCPP_WARN(get_logger(), "Empty path received");
      return;
    }
    path_ = msg;
    wp_idx_ = std::min(start_wp_idx_, msg->poses.size()-1);   // start at requested WP (e.g., #2)
    phase_ = Phase::ALIGN;                                    // always align first
    settle_start_ = now();
    RCLCPP_INFO(get_logger(), "Path with %zu WPs (frame=%s). Starting at WP %zu.",
                msg->poses.size(), msg->header.frame_id.c_str(), wp_idx_+1);
  }

  void onOdom(const nav_msgs::msg::Odometry::SharedPtr msg){ odom_ = msg; }

  void tick(){
    if (!path_ || !odom_) return;
    if (wp_idx_ >= path_->poses.size()){
      if (phase_ != Phase::DONE) { stop(); phase_ = Phase::DONE; RCLCPP_INFO(get_logger(), "Path done."); }
      return;
    }

    // current robot pose (odom)
    const double rx = odom_->pose.pose.position.x;
    const double ry = odom_->pose.pose.position.y;
    const double ryaw = yawOf(odom_->pose.pose.orientation);

    // current waypoint in odom
    const auto& wp_map = path_->poses[wp_idx_];
    geometry_msgs::msg::PoseStamped wp_odom;
    if (!mapToOdom(wp_map, wp_odom)) { stop(); return; }

    const double tx = wp_odom.pose.position.x;
    const double ty = wp_odom.pose.position.y;

    const double dx = tx - rx;
    const double dy = ty - ry;
    const double dist = std::hypot(dx, dy);
    const double yaw_des = std::atan2(dy, dx);
    const double yaw_err = angNorm(yaw_des - ryaw);

    // reached?
    if (dist <= position_tolerance_) {
      stop();
      ++wp_idx_;
      if (wp_idx_ >= path_->poses.size()) { phase_ = Phase::DONE; RCLCPP_INFO(get_logger(), "All waypoints reached."); return; }
      phase_ = Phase::ALIGN;               // next waypoint: align then drive
      settle_start_ = now();
      RCLCPP_INFO(get_logger(), "Reached WP %zu. Next: WP %zu.", wp_idx_, wp_idx_+1);
      return;
    }

    // logs
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 500,
      "State:%s  WP %zu/%zu  Robot:(%.2f,%.2f,%.1f°)  Target:(%.2f,%.2f)  Dist:%.2fm  YawErr:%.1f°",
      phaseName(phase_), wp_idx_+1, path_->poses.size(),
      rx, ry, ryaw*180.0/M_PI, tx, ty, dist, yaw_err*180.0/M_PI);

    // FSM with fixed speeds
    switch (phase_) {
      case Phase::ALIGN:
      {
        if (std::fabs(yaw_err) > align_tolerance_) {
          const double w = (yaw_err >= 0.0) ? +angular_speed_ : -angular_speed_;
          wheels(0.0, w);                      // pure rotate
          settle_start_ = now();               // refresh settle while moving
        } else {
          stop();
          if ((now() - settle_start_).seconds() >= settle_time_s_) {
            phase_ = Phase::DRIVE;
            RCLCPP_INFO(get_logger(), "Aligned to WP %zu. Driving straight...", wp_idx_+1);
          }
        }
        break;
      }

      case Phase::DRIVE:
      {
        // No drift check: just go straight
        wheels(linear_speed_, 0.0);
        // when dist <= position_tolerance_ we’ll stop/advance above
        break;
      }

      case Phase::SETTLE:   // (not used but kept for completeness)
        if ((now() - settle_start_).seconds() >= settle_time_s_) phase_ = Phase::ALIGN;
        break;

      case Phase::IDLE:
      case Phase::DONE:
      default:
        stop();
        break;
    }
  }

  const char* phaseName(Phase p) const {
    switch(p){
      case Phase::IDLE: return "IDLE";
      case Phase::ALIGN: return "ALIGN";
      case Phase::DRIVE: return "MOVING";
      case Phase::SETTLE: return "SETTLING";
      case Phase::DONE: return "DONE";
    }
    return "?";
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<HamrPointToPoint>());
  rclcpp::shutdown();
  return 0;
}
