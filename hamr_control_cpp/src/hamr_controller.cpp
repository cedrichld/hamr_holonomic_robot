#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"

#include "std_msgs/msg/float64.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "tf2_msgs/msg/tf_message.hpp"
#include "geometry_msgs/msg/pose_with_covariance.hpp"
#include "geometry_msgs/msg/quaternion.hpp"

#include "hamr_interfaces/msg/live_gains.hpp"
#include "hamr_interfaces/msg/reference_traj.hpp"

#include <Eigen/Dense>

using std::placeholders::_1;

static inline double wrapAngle(double a) {
  double b = std::fmod(a + M_PI, 2.0 * M_PI);
  if (b < 0.0) b += 2.0 * M_PI;
  return b - M_PI;
}

static inline double quatToYaw(const geometry_msgs::msg::Quaternion &q) {
  const double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
  const double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
  return std::atan2(siny_cosp, cosy_cosp);
}

struct PIDGains {
  double P{0.0}, I{0.0}, D{0.0};
};

class PIAccumulator {
public:
  explicit PIAccumulator(double limit) : limit_(std::abs(limit)) {}
  double update(double error, double dt) {
    sum_ += error * dt;
    sum_ = std::clamp(sum_, -limit_, limit_);
    return sum_;
  }
  void reset() { sum_ = 0.0; }
private:
  double sum_{0.0};
  double limit_{0.0};
};

class HamrControlNode : public rclcpp::Node {
public:
  HamrControlNode() : rclcpp::Node("hamr_controller_node") {
    // --- Parameters ---
    r_wheel_ = this->declare_parameter<double>("r_wheel", 0.0762);
    a_wheel_ = this->declare_parameter<double>("a_wheel", 0.149556);
    b_wheel_ = this->declare_parameter<double>("b_wheel", 0.19682);

    gains_x_.P   = this->declare_parameter<double>("P_x",   0.1);
    gains_x_.I   = this->declare_parameter<double>("I_x",   0.005);
    gains_x_.D   = this->declare_parameter<double>("D_x",   0.001);
    gains_y_.P   = this->declare_parameter<double>("P_y",   0.1);
    gains_y_.I   = this->declare_parameter<double>("I_y",   0.005);
    gains_y_.D   = this->declare_parameter<double>("D_y",   0.001);
    gains_yaw_.P = this->declare_parameter<double>("P_yaw", 0.5);
    gains_yaw_.I = this->declare_parameter<double>("I_yaw", 0.001);
    gains_yaw_.D = this->declare_parameter<double>("D_yaw", 0.001);

    control_rate_hz_ = this->declare_parameter<double>("control_rate_hz", 100.0);
    d_alpha_         = this->declare_parameter<double>("d_alpha", 0.4);

    param_cb_handle_ = this->add_on_set_parameters_callback(
      std::bind(&HamrControlNode::onParametersSet, this, _1));

    // --- Pub/Sub ---
    left_pub_   = this->create_publisher<std_msgs::msg::Float64>("/left_wheel/cmd_vel", 1);
    right_pub_  = this->create_publisher<std_msgs::msg::Float64>("/right_wheel/cmd_vel", 1);
    turret_pub_ = this->create_publisher<std_msgs::msg::Float64>("/turret/cmd_vel", 1);

    gains_pub_  = this->create_publisher<hamr_interfaces::msg::LiveGains>("/live_gains", 10);

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/hamr/odom", 1, std::bind(&HamrControlNode::odomCallback, this, _1));
    tf_sub_ = this->create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf", 10, std::bind(&HamrControlNode::tfCallback, this, _1));
    ref_sub_ = this->create_subscription<hamr_interfaces::msg::ReferenceTraj>(
      "/reference_trajectory", 1, std::bind(&HamrControlNode::referenceCallback, this, _1));

    // --- Timer ---
    last_control_time_ = this->now();
    const double period = 1.0 / std::max(1e-3, control_rate_hz_);
    control_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(period),
      std::bind(&HamrControlNode::controlTick, this));

    // Integrators and thresholds
    I_x_   = std::make_unique<PIAccumulator>(0.5);
    I_y_   = std::make_unique<PIAccumulator>(0.5);
    I_yaw_ = std::make_unique<PIAccumulator>(1.0);

    threshold_xy_  = 0.01;  // m
    threshold_yaw_ = 0.02;  // rad

    xy_dot_limit_  = 5.0;
    yaw_dot_limit_ = 2.0;

    RCLCPP_INFO(this->get_logger(),
      "HAMR Controller started: Px=%.3f Ix=%.3f Dx=%.3f; "
      "Py=%.3f Iy=%.3f Dy=%.3f; "
      "Pyaw=%.3f Iyaw=%.3f Dyaw=%.3f",
      gains_x_.P, gains_x_.I, gains_x_.D,
      gains_y_.P, gains_y_.I, gains_y_.D,
      gains_yaw_.P, gains_yaw_.I, gains_yaw_.D);
  }

private:
  // --- Callbacks ---
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    pose_base_ = *msg;   // store full odom (we use pose)
    have_pose_ = true;
  }

  void tfCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
    for (const auto &t : msg->transforms) {
      if (t.child_frame_id == "turret_link" && t.header.frame_id == "base_link") {
        turret_to_base_q_ = t.transform.rotation;
        have_turret_q_ = true;
        break;
      }
    }
  }

  void referenceCallback(const hamr_interfaces::msg::ReferenceTraj::SharedPtr msg) {
    reference_ = *msg;
    have_reference_ = true;
  }

  void controlTick() {
    const rclcpp::Time now = this->now();
    const double dt = std::max(1e-4, std::min((now - last_control_time_).seconds(), 0.1));
    last_control_time_ = now;

    if (!have_pose_ || !have_reference_ || !have_turret_q_) return;

    pidStep(dt);
  }

  // --- Core control ---
  void pidStep(double dt) {
    // Errors
    double err_x, err_y, err_yaw, yaw_base_w;
    std::tie(err_x, err_y, err_yaw, yaw_base_w) = computeErrors();

    // For /live_gains debug
    double P_x=0, I_x_term=0, D_x=0;
    double P_y=0, I_y_term=0, D_y=0;
    double P_yaw=0, I_yaw_term=0, D_yaw=0;

    // X loop
    double desired_x_dot = reference_.x_dot;
    if (std::abs(err_x) < threshold_xy_) {
      err_x_prev_ = 0.0;
      I_x_->reset();
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "RESET I_x at target x=%.3f", reference_.x);
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500, "X not at target: err=%.4f", err_x);
      P_x = gains_x_.P * err_x;
      I_x_term = gains_x_.I * I_x_->update(err_x, dt);

      const double d_raw_x = (err_x - err_x_prev_) / dt;
      d_err_x_filt_ = d_alpha_ * d_raw_x + (1.0 - d_alpha_) * d_err_x_filt_;
      D_x = gains_x_.D * d_err_x_filt_;

      desired_x_dot = reference_.x_dot + P_x + I_x_term + D_x;
      err_x_prev_ = err_x;
    }

    // Y loop
    double desired_y_dot = reference_.y_dot;
    if (std::abs(err_y) < threshold_xy_) {
      err_y_prev_ = 0.0;
      I_y_->reset();
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "RESET I_y at target y=%.3f", reference_.y);
    } else {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500, "Y not at target: err=%.4f", err_y);
      P_y = gains_y_.P * err_y;
      I_y_term = gains_y_.I * I_y_->update(err_y, dt);

      const double d_raw_y = (err_y - err_y_prev_) / dt;
      d_err_y_filt_ = d_alpha_ * d_raw_y + (1.0 - d_alpha_) * d_err_y_filt_;
      D_y = gains_y_.D * d_err_y_filt_;

      desired_y_dot = reference_.y_dot + P_y + I_y_term + D_y;
      err_y_prev_ = err_y;
    }

    // Cap XY speed norm
    double desired_xy_norm = std::hypot(desired_x_dot, desired_y_dot);
    if (desired_xy_norm > xy_dot_limit_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 500,
                           "CAPPING x,y velocity from %.3f to %.3f", desired_xy_norm, xy_dot_limit_);
      const double scale = xy_dot_limit_ / desired_xy_norm;
      desired_x_dot *= scale;
      desired_y_dot *= scale;
    }

    // Yaw loop
    double desired_yaw_dot = reference_.yaw_dot;
    if (std::abs(err_yaw) < threshold_yaw_) {
      err_yaw_prev_ = 0.0;
      I_yaw_->reset();
    } else {
      P_yaw = gains_yaw_.P * err_yaw;
      I_yaw_term = gains_yaw_.I * I_yaw_->update(err_yaw, dt);

      const double d_raw_yaw = (err_yaw - err_yaw_prev_) / dt;
      d_err_yaw_filt_ = d_alpha_ * d_raw_yaw + (1.0 - d_alpha_) * d_err_yaw_filt_;
      D_yaw = gains_yaw_.D * d_err_yaw_filt_;

      desired_yaw_dot = std::clamp(reference_.yaw_dot + P_yaw + I_yaw_term + D_yaw,
                                   -yaw_dot_limit_, yaw_dot_limit_);
      err_yaw_prev_ = err_yaw;
    }

    publishLiveGains(P_x, D_x, I_x_term, P_y, D_y, I_y_term, P_yaw, D_yaw, I_yaw_term);

    Eigen::Vector3d v;
    v << desired_x_dot, desired_y_dot, desired_yaw_dot;
    publishJointCmd(v, yaw_base_w);
  }

  std::tuple<double,double,double,double> computeErrors() {
    // Desired pose
    const double x_des = reference_.x;
    const double y_des = reference_.y;
    const double yaw_des = reference_.yaw;

    // Current base pose
    const auto &pose = pose_base_.pose.pose; // PoseWithCovariance → Pose
    const double x = pose.position.x;
    const double y = pose.position.y;
    const double yaw_base_w = quatToYaw(pose.orientation);

    // Turret orientation (base frame)
    const double yaw_turret_b = quatToYaw(turret_to_base_q_);
    const double yaw_turret_w = wrapAngle(yaw_base_w + yaw_turret_b);

    const double err_x = x_des - x;
    const double err_y = y_des - y;
    const double err_yaw = wrapAngle(yaw_des - yaw_turret_w);

    return {err_x, err_y, err_yaw, yaw_base_w};
  }

  void publishLiveGains(double Px, double Dx, double Ix,
                        double Py, double Dy, double Iy,
                        double Pyaw, double Dyaw, double Iyaw) {
    hamr_interfaces::msg::LiveGains g;
    g.p_x = Px; g.d_x = Dx; g.i_x = Ix;
    g.p_y = Py; g.d_y = Dy; g.i_y = Iy;
    g.p_yaw = Pyaw; g.d_yaw = Dyaw; g.i_yaw = Iyaw;
    gains_pub_->publish(g);
  }

  Eigen::Vector3d computeJointOmegas(const Eigen::Vector3d &desired_velocity, double yaw) const {
    const double r = r_wheel_, a = a_wheel_, b = b_wheel_;
    const double c = std::cos(yaw), s = std::sin(yaw);

    Eigen::Matrix3d J;
    // Matches your Python:
    // [ r/2*(c - s*b/a), r/2*(c + s*b/a), 0 ]
    // [ r/2*(s + c*b/a), r/2*(s - c*b/a), 0 ]
    // [ r/(2*a),        -r/(2*a),         1 ]
    J(0,0) = r * 0.5 * (c - s * b / a);
    J(0,1) = r * 0.5 * (c + s * b / a);
    J(0,2) = 0.0;

    J(1,0) = r * 0.5 * (s + c * b / a);
    J(1,1) = r * 0.5 * (s - c * b / a);
    J(1,2) = 0.0;

    J(2,0) =  r / (2.0 * a);
    J(2,1) = -r / (2.0 * a);
    J(2,2) =  1.0;

    // Solve J * omega = desired_velocity
    return J.colPivHouseholderQr().solve(desired_velocity);
  }

  void publishJointCmd(const Eigen::Vector3d &desired_velocity, double yaw) {
    const Eigen::Vector3d omegas = computeJointOmegas(desired_velocity, yaw);

    std_msgs::msg::Float64 right_msg, left_msg, turret_msg;
    right_msg.data  = omegas(0);
    left_msg.data   = omegas(1);
    turret_msg.data = omegas(2);

    right_pub_->publish(right_msg);
    left_pub_->publish(left_msg);
    turret_pub_->publish(turret_msg);
  }

  // --- Parameters callback ---
  rcl_interfaces::msg::SetParametersResult
  onParametersSet(const std::vector<rclcpp::Parameter> &params) {
    for (const auto &p : params) {
      const std::string &name = p.get_name();
      if      (name == "P_x")    gains_x_.P = p.as_double();
      else if (name == "I_x")    gains_x_.I = p.as_double();
      else if (name == "D_x")    gains_x_.D = p.as_double();
      else if (name == "P_y")    gains_y_.P = p.as_double();
      else if (name == "I_y")    gains_y_.I = p.as_double();
      else if (name == "D_y")    gains_y_.D = p.as_double();
      else if (name == "P_yaw")  gains_yaw_.P = p.as_double();
      else if (name == "I_yaw")  gains_yaw_.I = p.as_double();
      else if (name == "D_yaw")  gains_yaw_.D = p.as_double();
      else if (name == "r_wheel")   r_wheel_ = p.as_double();
      else if (name == "a_wheel")   a_wheel_ = p.as_double();
      else if (name == "b_wheel")   b_wheel_ = p.as_double();
      else if (name == "control_rate_hz") {
        control_rate_hz_ = p.as_double();
        const double period = 1.0 / std::max(1e-3, control_rate_hz_);
        control_timer_->cancel();
        control_timer_ = this->create_wall_timer(
          std::chrono::duration<double>(period),
          std::bind(&HamrControlNode::controlTick, this));
      }
      else if (name == "d_alpha")   d_alpha_ = p.as_double();
    }
    rcl_interfaces::msg::SetParametersResult res;
    res.successful = true;
    return res;
  }

private:
  // Params
  double r_wheel_{0.0762}, a_wheel_{0.149556}, b_wheel_{0.19682};
  PIDGains gains_x_, gains_y_, gains_yaw_;
  double control_rate_hz_{100.0}, d_alpha_{0.4};

  // Pub/Sub
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_pub_, right_pub_, turret_pub_;
  rclcpp::Publisher<hamr_interfaces::msg::LiveGains>::SharedPtr gains_pub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_;
  rclcpp::Subscription<hamr_interfaces::msg::ReferenceTraj>::SharedPtr ref_sub_;

  // Timer/Timing
  rclcpp::TimerBase::SharedPtr control_timer_;
  rclcpp::Time last_control_time_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_cb_handle_;

  // State
  nav_msgs::msg::Odometry pose_base_;
  geometry_msgs::msg::Quaternion turret_to_base_q_;
  hamr_interfaces::msg::ReferenceTraj reference_;
  bool have_pose_{false}, have_turret_q_{false}, have_reference_{false};

  // PID state
  std::unique_ptr<PIAccumulator> I_x_, I_y_, I_yaw_;
  double err_x_prev_{0.0}, err_y_prev_{0.0}, err_yaw_prev_{0.0};
  double d_err_x_filt_{0.0}, d_err_y_filt_{0.0}, d_err_yaw_filt_{0.0};
  double threshold_xy_{0.01}, threshold_yaw_{0.02};
  double xy_dot_limit_{5.0}, yaw_dot_limit_{2.0};
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<HamrControlNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
