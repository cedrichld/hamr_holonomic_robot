#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>
#include <functional>

using namespace std;
using namespace std::chrono_literals;

struct TrajectoryPoint {
  double t;      // time
  double x, y;   // position
  double vx, vy; // velocity
  double theta;  // heading
  double w;      // angular velocity
  
  TrajectoryPoint(double t=0, double x=0, double y=0, double vx=0, double vy=0, double theta=0, double w=0)
    : t(t), x(x), y(y), vx(vx), vy(vy), theta(theta), w(w) {}
};

class TrajectoryGenerator {
public:
  // Generate different trajectory types
  static vector<TrajectoryPoint> generateSquare(double side_length, double speed, double dt = 0.02) {
    vector<TrajectoryPoint> traj;
    double total_time = 4 * side_length / speed + 4 * (M_PI/2) / 1.0; // straight time + turn time
    
    double t = 0;
    double x = 0, y = 0, theta = 0;
    
    // More sophisticated square with smooth corners
    for (int side = 0; side < 4; side++) {
      // Straight segment
      double straight_time = side_length / speed;
      for (double dt_seg = 0; dt_seg < straight_time; dt_seg += dt) {
        double progress = dt_seg / straight_time;
        traj.emplace_back(t, 
                         x + progress * side_length * cos(theta),
                         y + progress * side_length * sin(theta),
                         speed * cos(theta), speed * sin(theta),
                         theta, 0);
        t += dt;
      }
      
      // Update position after straight segment
      x += side_length * cos(theta);
      y += side_length * sin(theta);
      
      // Corner turn (90 degrees)
      double turn_time = (M_PI/2) / 1.0; // 1 rad/s turn rate
      for (double dt_seg = 0; dt_seg < turn_time; dt_seg += dt) {
        double turn_progress = dt_seg / turn_time;
        double current_theta = theta + turn_progress * (M_PI/2);
        traj.emplace_back(t, x, y, 0, 0, current_theta, 1.0);
        t += dt;
      }
      
      theta += M_PI/2;
      if (theta > M_PI) theta -= 2*M_PI;
    }
    
    return traj;
  }
  
  static vector<TrajectoryPoint> generateCircle(double radius, double angular_speed, double dt = 0.02) {
    vector<TrajectoryPoint> traj;
    double period = 2 * M_PI / angular_speed;
    
    for (double t = 0; t < period; t += dt) {
      double theta = angular_speed * t;
      double x = radius * cos(theta);
      double y = radius * sin(theta);
      double vx = -radius * angular_speed * sin(theta);
      double vy = radius * angular_speed * cos(theta);
      double heading = theta + M_PI/2; // tangent direction
      
      traj.emplace_back(t, x, y, vx, vy, heading, angular_speed);
    }
    
    return traj;
  }
  
  static vector<TrajectoryPoint> generateFigureEight(double radius, double angular_speed, double dt = 0.02) {
    vector<TrajectoryPoint> traj;
    double period = 4 * M_PI / angular_speed; // Two loops
    
    for (double t = 0; t < period; t += dt) {
      double s = angular_speed * t;
      double x = radius * sin(s);
      double y = radius * sin(s) * cos(s); // figure-8 parametrization
      
      double vx = radius * angular_speed * cos(s);
      double vy = radius * angular_speed * (cos(2*s));
      
      double heading = atan2(vy, vx);
      double w = angular_speed * sin(s); // varying angular velocity
      
      traj.emplace_back(t, x, y, vx, vy, heading, w);
    }
    
    return traj;
  }
  
  static vector<TrajectoryPoint> generateSinusoidalPath(double amplitude, double frequency, double forward_speed, double length, double dt = 0.02) {
    vector<TrajectoryPoint> traj;
    double total_time = length / forward_speed;
    
    for (double t = 0; t < total_time; t += dt) {
      double x = forward_speed * t;
      double y = amplitude * sin(2 * M_PI * frequency * t);
      
      double vx = forward_speed;
      double vy = amplitude * 2 * M_PI * frequency * cos(2 * M_PI * frequency * t);
      
      double heading = atan2(vy, vx);
      double curvature = (amplitude * pow(2 * M_PI * frequency, 2) * sin(2 * M_PI * frequency * t)) / 
                        pow(sqrt(vx*vx + vy*vy), 3);
      double w = curvature * sqrt(vx*vx + vy*vy);
      
      traj.emplace_back(t, x, y, vx, vy, heading, w);
    }
    
    return traj;
  }
};

static double normalize_angle(double a) {
  while (a >  M_PI) a -= 2.0 * M_PI;
  while (a <= -M_PI) a += 2.0 * M_PI;
  return a;
}

class TrajectoryFollowerNode : public rclcpp::Node {
public:
  TrajectoryFollowerNode()
  : Node("trajectory_follower_node"),
    wheel_base_(declare_parameter("wheel_base_m", 0.16)),
    max_linear_vel_(declare_parameter("max_linear_vel", 2.0)),
    max_angular_vel_(declare_parameter("max_angular_vel", 3.0)),
    // MPC-style gains
    kp_x_(declare_parameter("kp_x", 2.0)),
    kp_y_(declare_parameter("kp_y", 2.0)), 
    kp_theta_(declare_parameter("kp_theta", 3.0)),
    kd_x_(declare_parameter("kd_x", 0.5)),
    kd_y_(declare_parameter("kd_y", 0.5)),
    kd_theta_(declare_parameter("kd_theta", 0.2)),
    // Feedforward gains
    ff_linear_(declare_parameter("ff_linear", 0.8)),
    ff_angular_(declare_parameter("ff_angular", 0.7)),
    lookahead_time_(declare_parameter("lookahead_time", 0.1)),
    trajectory_type_(declare_parameter("trajectory_type", "square"))
  {
    left_pub_  = create_publisher<std_msgs::msg::Float64>("/left_wheel/cmd_vel", 1);
    right_pub_ = create_publisher<std_msgs::msg::Float64>("/right_wheel/cmd_vel", 1);

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/hamr/odom", 40,
      [this](const nav_msgs::msg::Odometry::SharedPtr m){ this->on_odom(m); });

    timer_ = create_wall_timer(20ms, [this]{ this->control_update(); });
    
    generateTrajectory();
    RCLCPP_INFO(this->get_logger(), "Trajectory Follower initialized with %s trajectory (%zu points)", 
                trajectory_type_.c_str(), trajectory_.size());
  }

private:
  void generateTrajectory() {
    // Generate different trajectories based on parameter
    if (trajectory_type_ == "square") {
      trajectory_ = TrajectoryGenerator::generateSquare(3.0, 1.0);
    } else if (trajectory_type_ == "circle") {
      trajectory_ = TrajectoryGenerator::generateCircle(1.5, 0.5);
    } else if (trajectory_type_ == "figure8") {
      trajectory_ = TrajectoryGenerator::generateFigureEight(2.0, 0.3);
    } else if (trajectory_type_ == "sine") {
      trajectory_ = TrajectoryGenerator::generateSinusoidalPath(1.0, 0.5, 0.8, 10.0);
    } else {
      // Default to square
      trajectory_ = TrajectoryGenerator::generateSquare(3.0, 1.0);
    }
    
    if (!trajectory_.empty()) {
      start_time_ = this->now();
      trajectory_started_ = false;
    }
  }

  void on_odom(const nav_msgs::msg::Odometry::SharedPtr &msg) {
    current_x_ = msg->pose.pose.position.x;
    current_y_ = msg->pose.pose.position.y;

    const auto &q = msg->pose.pose.orientation;
    const double s = 2.0 * (q.w * q.z + q.x * q.y);
    const double c = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    current_theta_ = std::atan2(s, c);
    
    current_vx_ = msg->twist.twist.linear.x;
    current_vy_ = msg->twist.twist.linear.y;
    current_w_ = msg->twist.twist.angular.z;

    have_odom_ = true;
    
    if (!trajectory_started_ && have_odom_) {
      // Offset trajectory to start from current position
      double offset_x = current_x_ - (trajectory_.empty() ? 0 : trajectory_[0].x);
      double offset_y = current_y_ - (trajectory_.empty() ? 0 : trajectory_[0].y);
      
      for (auto& pt : trajectory_) {
        pt.x += offset_x;
        pt.y += offset_y;
      }
      
      trajectory_started_ = true;
      start_time_ = this->now();
      RCLCPP_INFO(this->get_logger(), "Trajectory started from (%.2f, %.2f)", current_x_, current_y_);
    }
  }

  TrajectoryPoint getDesiredState(double current_time) {
    if (trajectory_.empty()) return TrajectoryPoint();
    
    // Find closest trajectory point by time
    double trajectory_time = current_time;
    
    // Handle trajectory looping
    if (!trajectory_.empty()) {
      double max_time = trajectory_.back().t;
      if (trajectory_time > max_time) {
        trajectory_time = fmod(trajectory_time, max_time);
      }
    }
    
    // Find the trajectory segment we're in
    size_t idx = 0;
    for (size_t i = 0; i < trajectory_.size() - 1; i++) {
      if (trajectory_[i].t <= trajectory_time && trajectory_time < trajectory_[i+1].t) {
        idx = i;
        break;
      }
    }
    
    // Linear interpolation between trajectory points
    if (idx < trajectory_.size() - 1) {
      const auto& p1 = trajectory_[idx];
      const auto& p2 = trajectory_[idx + 1];
      double alpha = (trajectory_time - p1.t) / (p2.t - p1.t);
      
      TrajectoryPoint desired;
      desired.x = p1.x + alpha * (p2.x - p1.x);
      desired.y = p1.y + alpha * (p2.y - p1.y);
      desired.vx = p1.vx + alpha * (p2.vx - p1.vx);
      desired.vy = p1.vy + alpha * (p2.vy - p1.vy);
      desired.theta = p1.theta + alpha * normalize_angle(p2.theta - p1.theta);
      desired.w = p1.w + alpha * (p2.w - p1.w);
      
      return desired;
    }
    
    return trajectory_.back();
  }

  void control_update() {
    if (!have_odom_ || !trajectory_started_ || trajectory_.empty()) {
      publish_zero();
      return;
    }

    double current_time = (this->now() - start_time_).seconds();
    
    // Get desired state (with lookahead for better tracking)
    TrajectoryPoint desired = getDesiredState(current_time + lookahead_time_);
    
    // Calculate errors
    double ex = desired.x - current_x_;
    double ey = desired.y - current_y_;
    double etheta = normalize_angle(desired.theta - current_theta_);
    
    // Velocity errors for derivative term
    double evx = desired.vx - current_vx_;
    double evy = desired.vy - current_vy_;
    double ew = desired.w - current_w_;
    
    // Transform position errors to robot frame
    double ex_robot = ex * cos(current_theta_) + ey * sin(current_theta_);
    double ey_robot = -ex * sin(current_theta_) + ey * cos(current_theta_);
    
    // Control law: Feedforward + Feedback
    double v_cmd = desired.vx * cos(current_theta_) + desired.vy * sin(current_theta_); // feedforward
    v_cmd += kp_x_ * ex_robot + kd_x_ * evx; // feedback
    
    double w_cmd = ff_angular_ * desired.w; // feedforward
    w_cmd += kp_theta_ * etheta + kd_theta_ * ew; // feedback
    w_cmd += kp_y_ * ey_robot; // lateral error correction
    
    // Apply limits
    v_cmd = std::clamp(v_cmd, -max_linear_vel_, max_linear_vel_);
    w_cmd = std::clamp(w_cmd, -max_angular_vel_, max_angular_vel_);
    
    // Convert to differential drive
    convert_to_wheel_speeds(v_cmd, w_cmd);
    
    // Debug output (every 50 cycles = 1Hz)
    if (debug_counter_++ % 50 == 0) {
      RCLCPP_INFO(this->get_logger(), 
                  "t=%.2f, pos=(%.2f,%.2f)->( %.2f,%.2f), err=(%.3f,%.3f,%.3f°)", 
                  current_time, current_x_, current_y_, desired.x, desired.y,
                  ex, ey, etheta * 180.0 / M_PI);
    }
  }

  void convert_to_wheel_speeds(double v, double w) {
    double v_left = v - w * wheel_base_ / 2.0;
    double v_right = v + w * wheel_base_ / 2.0;
    
    std_msgs::msg::Float64 L, R;
    L.data = v_left;
    R.data = v_right;
    
    left_pub_->publish(L);
    right_pub_->publish(R);
  }

  void publish_zero() {
    std_msgs::msg::Float64 L, R;
    L.data = R.data = 0.0;
    left_pub_->publish(L);
    right_pub_->publish(R);
  }

  // ROS interfaces
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_pub_, right_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Parameters
  double wheel_base_;
  double max_linear_vel_, max_angular_vel_;
  double kp_x_, kp_y_, kp_theta_;
  double kd_x_, kd_y_, kd_theta_;
  double ff_linear_, ff_angular_;
  double lookahead_time_;
  std::string trajectory_type_;

  // State
  bool have_odom_{false};
  bool trajectory_started_{false};
  double current_x_{0}, current_y_{0}, current_theta_{0};
  double current_vx_{0}, current_vy_{0}, current_w_{0};
  rclcpp::Time start_time_;
  int debug_counter_{0};

  // Trajectory
  std::vector<TrajectoryPoint> trajectory_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrajectoryFollowerNode>());
  rclcpp::shutdown();
  return 0;
}