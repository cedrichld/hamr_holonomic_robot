#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>

#include <linux/joystick.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#include <atomic>
#include <cmath>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <algorithm>

using std::placeholders::_1;

class Teleoperation : public rclcpp::Node {
public:
  Teleoperation()
  : Node("teleop_node")
  {
    // ----- Parameters (override in a YAML or on the command line) -----
    device_           = this->declare_parameter<std::string>("device", "/dev/input/js0");
    axis_ly_          = this->declare_parameter<int>("axis_ly", 1);   // Left stick Y
    axis_rx_          = this->declare_parameter<int>("axis_rx", 3);   // Right stick X
    axis_lt_          = this->declare_parameter<int>("axis_lt", 2);   // Left trigger
    axis_rt_          = this->declare_parameter<int>("axis_rt", 5);   // Right trigger
    invert_ly_        = this->declare_parameter<bool>("invert_ly", true);   // up is usually -1
    invert_rx_        = this->declare_parameter<bool>("invert_rx", false);
    invert_lt_        = this->declare_parameter<bool>("invert_lt", false);
    invert_rt_        = this->declare_parameter<bool>("invert_rt", false);
    deadman_button_   = this->declare_parameter<int>("deadman_button", -1); // -1 disables
    deadband_         = this->declare_parameter<double>("deadband", 0.08);
    mix_scale_        = this->declare_parameter<double>("mix_scale", 0.8);  // your 0.8 factor
    max_rpm_cmd_      = this->declare_parameter<double>("max_rpm_cmd", 30.0);
    turret_max_rad_s_ = this->declare_parameter<double>("turret_max_rad_s", 2.0);
    publish_rate_hz_  = this->declare_parameter<double>("publish_rate_hz", 100.0);
    stale_timeout_s_  = this->declare_parameter<double>("stale_timeout_s", 100.0);
    ema_tau_s_        = this->declare_parameter<double>("ema_tau_s", 0.05); // output smoothing

    // Convert max RPM to rad/s
    const double RPM_TO_RAD_S = 2.0 * M_PI / 60.0;
    max_rad_s_ = max_rpm_cmd_ * RPM_TO_RAD_S;

    // ----- Publishers -----
    auto qos = rclcpp::QoS(1).best_effort();
    pub_left_   = this->create_publisher<std_msgs::msg::Float64>("/left_wheel/cmd_vel",  qos);
    pub_right_  = this->create_publisher<std_msgs::msg::Float64>("/right_wheel/cmd_vel", qos);
    pub_turret_ = this->create_publisher<std_msgs::msg::Float64>("/turret/cmd_vel",      qos);

    // ----- Start joystick reader thread -----
    running_ = true;
    reader_ = std::thread([this]{ this->reader_loop(); });

    // ----- Timer to publish at fixed rate -----
    timer_ = this->create_wall_timer(
      std::chrono::microseconds((int)(1e6 / std::max(1.0, publish_rate_hz_))),
      std::bind(&Teleoperation::on_timer, this));

    RCLCPP_INFO(this->get_logger(),
      "teleop_node ready (device=%s, max_rpm=%.1f → max_rad_s=%.3f)",
      device_.c_str(), max_rpm_cmd_, max_rad_s_);
  }

  ~Teleoperation() override {
    running_ = false;
    if (reader_.joinable()) reader_.join();
    if (fd_ >= 0) ::close(fd_);
  }

private:
  // Normalize int16 joystick axis to [-1, 1]
  static inline float norm_axis(int16_t v) {
    // js_event value is -32767..32767
    float x = (v >= 0) ? (float)v / 32767.0f : (float)v / 32768.0f;
    if (x > 1.f) x = 1.f;
    if (x < -1.f) x = -1.f;
    return x;
  }

  // Apply deadband and optional inversion
  inline float shape(float x, bool invert) const {
    if (invert) x = -x;
    if (std::fabs(x) < deadband_) return 0.0f;
    // rescale to keep full-range after deadband
    float s = (std::fabs(x) - (float)deadband_) / (1.0f - (float)deadband_);
    if (s < 0.f) s = 0.f; 
    if (s > 1.f) s = 1.f;
    return std::copysign(s, x);
  }

  void reader_loop() {
    // Pre-size axis/button arrays
    axes_.assign(8, 0.0f);
    buttons_.assign(16, 0);
    last_event_time_ = now_steady();

    while (running_) {
      // Open (or reopen) joystick
      if (fd_ < 0) {
        fd_ = ::open(device_.c_str(), O_RDONLY | O_NONBLOCK);
        if (fd_ < 0) {
          RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                               "Cannot open %s, retrying...", device_.c_str());
          std::this_thread::sleep_for(std::chrono::milliseconds(200));
          continue;
        }
        // Query counts to size arrays properly
        unsigned char naxes = 0, nbuttons = 0;
        ioctl(fd_, JSIOCGAXES, &naxes);
        ioctl(fd_, JSIOCGBUTTONS, &nbuttons);
        axes_.assign(std::max<int>(naxes, (int)axes_.size()), 0.0f);
        buttons_.assign(std::max<int>(nbuttons, (int)buttons_.size()), 0);
        RCLCPP_INFO(this->get_logger(), "Opened %s (axes=%d, buttons=%d)", device_.c_str(), (int)naxes, (int)nbuttons);
      }

      // Read all available events
      js_event e;
      ssize_t n = 0;
      bool got_any = false;
      while ((n = ::read(fd_, &e, sizeof(e))) > 0) {
        got_any = true;
        switch (e.type & ~JS_EVENT_INIT) {
          case JS_EVENT_AXIS:
            if (e.number < axes_.size()) {
              axes_[e.number] = norm_axis(e.value);
              last_event_time_ = now_steady();
            }
            break;
          case JS_EVENT_BUTTON:
            if (e.number < buttons_.size()) {
              buttons_[e.number] = (e.value ? 1 : 0);
              last_event_time_ = now_steady();
            }
            break;
        }
      }

      if (!got_any) {
        // Idle a bit; also handles EAGAIN
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      } else {
        // tight loop is fine if events stream in
      }

      // If device disappeared, close and retry
      if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        ::close(fd_); fd_ = -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
      }
    }
  }

  static inline double now_steady() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
  }

  void on_timer() {
    // Copy latest inputs atomically-ish
    std::vector<float> axes = axes_;     // small, fine to copy
    std::vector<uint8_t> btns = buttons_;

    // Stale/Deadman
    bool ok = false;
    const double tnow = now_steady();
    if ((tnow - last_event_time_) < stale_timeout_s_) {
      if (deadman_button_ < 0 || (deadman_button_ < (int)btns.size() && btns[deadman_button_])) {
        ok = true;
      }
    }

    // Defaults
    float ly = 0.f, rx = 0.f, lt = 0.f, rt = 0.f;

    if (ok) {
      if (axis_ly_ >= 0 && axis_ly_ < (int)axes.size()) ly = shape(axes[axis_ly_], invert_ly_);
      if (axis_rx_ >= 0 && axis_rx_ < (int)axes.size()) rx = shape(axes[axis_rx_], invert_rx_);
      if (axis_lt_ >= 0 && axis_lt_ < (int)axes.size()) lt = shape(axes[axis_lt_], invert_lt_);
      if (axis_rt_ >= 0 && axis_rt_ < (int)axes.size()) rt = shape(axes[axis_rt_], invert_rt_);
    }

    // Your mix (scale 0.8 on sticks)
    const float forward = ly * (float)mix_scale_;
    const float turn    = rx * (float)mix_scale_;

    // Left/right wheel commands in rad/s
    const double left_cmd  = (forward + turn) * max_rad_s_;
    const double right_cmd = (forward - turn) * max_rad_s_;

    // Turret: RT - LT, scaled to turret_max_rad_s_
    const double turret_cmd = (double)(rt - lt) * turret_max_rad_s_;

    // Simple EMA smoothing to reduce jitter
    const double dt = 1.0 / std::max(1.0, publish_rate_hz_);
    const double a  = (ema_tau_s_ > 1e-4) ? dt / (ema_tau_s_ + dt) : 1.0;

    left_f_   += a * (left_cmd   - left_f_);
    right_f_  += a * (right_cmd  - right_f_);
    turret_f_ += a * (turret_cmd - turret_f_);

    // Publish
    std_msgs::msg::Float64 m;
    m.data = left_f_;   pub_left_->publish(m);
    m.data = right_f_;  pub_right_->publish(m);
    m.data = turret_f_; pub_turret_->publish(m);
  }

private:
  // Params
  std::string device_;
  int axis_ly_, axis_rx_, axis_lt_, axis_rt_, deadman_button_;
  bool invert_ly_, invert_rx_, invert_lt_, invert_rt_;
  double deadband_, mix_scale_, max_rpm_cmd_, max_rad_s_, turret_max_rad_s_;
  double publish_rate_hz_, stale_timeout_s_, ema_tau_s_;

  // State
  std::vector<float>  axes_;
  std::vector<uint8_t> buttons_;
  std::atomic<bool> running_{false};
  std::thread reader_;
  int fd_{-1};
  double last_event_time_{0.0};
  rclcpp::TimerBase::SharedPtr timer_;

  // Smoothed outputs
  double left_f_{0.0}, right_f_{0.0}, turret_f_{0.0};

  // Publishers
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr pub_left_, pub_right_, pub_turret_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Teleoperation>());
  rclcpp::shutdown();
  return 0;
}
