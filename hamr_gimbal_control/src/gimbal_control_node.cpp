#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <std_msgs/msg/float32.hpp>

#include <sys/socket.h>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <optional>

using namespace std::chrono_literals;

static inline double deg2rad(double d){ return d * M_PI / 180.0; }
static inline double rad2deg(double r){ return r * 180.0 / M_PI; }
static inline double clamp(double v, double lo, double hi){ return std::max(lo, std::min(v, hi)); }

struct CanRx {
  int sock = -1;
  std::atomic<bool> running{false};
  std::thread th;

  bool open(const std::string& iface, const std::vector<can_filter>& filters) {
    sock = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (sock < 0) { perror("socket"); return false; }
    
    struct ifreq ifr{};
    std::strncpy(ifr.ifr_name, iface.c_str(), IFNAMSIZ);
    if (ioctl(sock, SIOCGIFINDEX, &ifr) < 0) { 
      perror("SIOCGIFINDEX"); 
      close(sock); 
      sock=-1; 
      return false; 
    }
    
    if (!filters.empty()) {
      if (setsockopt(sock, SOL_CAN_RAW, CAN_RAW_FILTER, filters.data(), 
                     filters.size()*sizeof(can_filter)) < 0) {
        perror("setsockopt FILTER");
      }
    }
    
    struct sockaddr_can addr{};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) { 
      perror("bind"); 
      close(sock); 
      sock=-1; 
      return false; 
    }
    return true;
  }
  
  void closeSock() { 
    if (sock>=0) { 
      close(sock); 
      sock=-1; 
    } 
  }
};

struct CanTx {
  int sock = -1;
  
  bool open(const std::string& iface) {
    sock = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (sock < 0) { perror("socket"); return false; }
    
    struct ifreq ifr{};
    std::strncpy(ifr.ifr_name, iface.c_str(), IFNAMSIZ);
    if (ioctl(sock, SIOCGIFINDEX, &ifr) < 0) { 
      perror("SIOCGIFINDEX"); 
      close(sock); 
      sock=-1; 
      return false; 
    }
    
    struct sockaddr_can addr{};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) { 
      perror("bind"); 
      close(sock); 
      sock=-1; 
      return false; 
    }
    return true;
  }
  
  bool sendFrame(uint32_t id, const uint8_t data[8], uint8_t dlc=8) {
    if (sock < 0) return false;
    struct can_frame f{};
    f.can_id = id;
    f.can_dlc = dlc;
    std::memcpy(f.data, data, dlc);
    int n = write(sock, &f, sizeof(f));
    return n == (int)sizeof(f);
  }
  
  void closeSock() { 
    if (sock>=0) { 
      close(sock); 
      sock=-1; 
    } 
  }
};

struct MotorState {
  int8_t temperature_c = 0;
  double current_A = 0.0;
  int16_t speed_dps = 0;
  int16_t angle_deg = 0;
  rclcpp::Time last_update;
  bool valid = false;
};

class GimbalControlNode : public rclcpp::Node {
public:
  GimbalControlNode() : Node("gimbal_control_node") {
    // Parameters
    can_iface_          = this->declare_parameter<std::string>("can_iface", "can0");
    imu_id_             = this->declare_parameter<int>("imu_id", 0x100);
    motor_roll_id_      = this->declare_parameter<int>("motor_roll_id", 2);
    motor_pitch_id_     = this->declare_parameter<int>("motor_pitch_id", 3);

    sigma_roll_         = this->declare_parameter<double>("sigma_roll",  +1.0);
    sigma_pitch_        = this->declare_parameter<double>("sigma_pitch", +1.0);
    q1_zero_            = this->declare_parameter<double>("q1_zero_rad", 0.0);
    q2_zero_            = this->declare_parameter<double>("q2_zero_rad", 0.0);

    enable_control_     = this->declare_parameter<bool>("enable_control", true);
    enable_can_command_ = this->declare_parameter<bool>("enable_can_command", false);
    rate_hz_            = this->declare_parameter<double>("rate_hz", 200.0);

    kp_roll_  = this->declare_parameter<double>("kp_roll",  2.0);
    kd_roll_  = this->declare_parameter<double>("kd_roll",  0.04);
    kp_pitch_ = this->declare_parameter<double>("kp_pitch", 2.0);
    kd_pitch_ = this->declare_parameter<double>("kd_pitch", 0.04);

    roll_ref_  = deg2rad(this->declare_parameter<double>("roll_ref_deg",  0.0));
    pitch_ref_ = deg2rad(this->declare_parameter<double>("pitch_ref_deg", 0.0));

    max_rad_        = deg2rad(this->declare_parameter<double>("max_deg", 25.0));
    slew_rad_per_s_ = deg2rad(this->declare_parameter<double>("slew_deg_per_s", 120.0));
    max_speed_dps_  = this->declare_parameter<int>("max_motor_speed_dps", 500);
    
    imu_timeout_s_   = this->declare_parameter<double>("imu_timeout_s", 0.5);
    release_brakes_  = this->declare_parameter<bool>("release_brakes_on_start", true);
    manual_timeout_s_ = this->declare_parameter<double>("manual_timeout_s", 1.0);

    // Publishers - radians (original)
    rpy_pub_     = this->create_publisher<geometry_msgs::msg::Vector3>("gimbal/imu_rpy_rad", 10);
    target_pub_  = this->create_publisher<std_msgs::msg::Float64MultiArray>("gimbal/motor_targets_rad", 10);
    
    // Publishers - degrees (new)
    rpy_deg_pub_    = this->create_publisher<geometry_msgs::msg::Vector3>("gimbal/imu_rpy_deg", 10);
    target_deg_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>("gimbal/motor_targets_deg", 10);
    
    temp_roll_pub_  = this->create_publisher<std_msgs::msg::Float32>("gimbal/motor_roll_temp", 10);
    temp_pitch_pub_ = this->create_publisher<std_msgs::msg::Float32>("gimbal/motor_pitch_temp", 10);

    // Subscriber for manual commands in degrees
    manual_cmd_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
      "gimbal/manual_command_deg", 10,
      [this](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        if (msg->data.size() >= 2) {
          std::lock_guard<std::mutex> lk(m_);
          manual_roll_cmd_rad_  = deg2rad(msg->data[0]);
          manual_pitch_cmd_rad_ = deg2rad(msg->data[1]);
          last_manual_time_ = this->now();
          manual_mode_ = true;
          RCLCPP_INFO(get_logger(), "Manual command: roll=%.2f° pitch=%.2f°", 
                      msg->data[0], msg->data[1]);
        }
      });

    // CAN setup
    setupCAN();
    
    // Optional: Release motor brakes
    if (release_brakes_ && enable_can_command_) {
      std::this_thread::sleep_for(500ms);
      releaseMotorBrakes();
    }

    // Start control loop
    if (enable_control_) {
      auto period = std::chrono::duration<double>(1.0 / rate_hz_);
      timer_ = this->create_wall_timer(
        std::chrono::duration_cast<std::chrono::nanoseconds>(period),
        std::bind(&GimbalControlNode::controlStep, this));
    }

    RCLCPP_INFO(get_logger(), 
                "Gimbal control started. CAN=%s IMU=0x%03X Roll=%d Pitch=%d Control=%s Commands=%s",
                can_iface_.c_str(), imu_id_, motor_roll_id_, motor_pitch_id_,
                enable_control_ ? "ON" : "OFF",
                enable_can_command_ ? "ON" : "OFF");
    RCLCPP_INFO(get_logger(), "Manual control: Subscribe to /gimbal/manual_command_deg [roll_deg, pitch_deg]");
  }

  ~GimbalControlNode() override {
    rx_.running = false;
    if (rx_.th.joinable()) rx_.th.join();
    rx_.closeSock();
    tx_.closeSock();
  }

private:
  void setupCAN() {
    std::vector<can_filter> filters;
    
    // IMU filter
    can_filter imu_f{};
    imu_f.can_id   = (uint32_t)imu_id_;
    imu_f.can_mask = CAN_SFF_MASK;
    filters.push_back(imu_f);
    
    // Motor reply filters (0x240 + motor_id)
    can_filter roll_f{};
    roll_f.can_id   = 0x240 + motor_roll_id_;
    roll_f.can_mask = CAN_SFF_MASK;
    filters.push_back(roll_f);
    
    can_filter pitch_f{};
    pitch_f.can_id   = 0x240 + motor_pitch_id_;
    pitch_f.can_mask = CAN_SFF_MASK;
    filters.push_back(pitch_f);

    if (!rx_.open(can_iface_, filters)) {
      RCLCPP_FATAL(get_logger(), "Failed to open CAN RX on %s", can_iface_.c_str());
      throw std::runtime_error("CAN RX open failed");
    }
    
    if (!tx_.open(can_iface_)) {
      RCLCPP_FATAL(get_logger(), "Failed to open CAN TX on %s", can_iface_.c_str());
      throw std::runtime_error("CAN TX open failed");
    }
    
    rx_.running = true;
    rx_.th = std::thread([this]{ this->canRxThread(); });
  }

  void canRxThread() {
    while (rclcpp::ok() && rx_.running) {
      struct can_frame f{};
      int n = read(rx_.sock, &f, sizeof(f));
      if (n < 0) {
        std::this_thread::sleep_for(1ms);
        continue;
      }
      if ((int)sizeof(f) != n) continue;
      if ((f.can_id & CAN_EFF_FLAG) != 0) continue;
      
      uint32_t id = f.can_id & CAN_SFF_MASK;
      
      // IMU data (custom format - adjust to your IMU)
      if (id == (uint32_t)imu_id_ && f.can_dlc >= 6) {
        // Assuming format: roll(16bit), pitch(16bit), yaw(16bit) in centidegrees
        int16_t r_cd = (int16_t)((f.data[1] << 8) | f.data[0]);
        int16_t p_cd = (int16_t)((f.data[3] << 8) | f.data[2]);
        int16_t y_cd = (int16_t)((f.data[5] << 8) | f.data[4]);
        
        constexpr double cd2rad = M_PI / 18000.0;
        double roll  = (double)r_cd * cd2rad;
        double pitch = (double)p_cd * cd2rad;
        double yaw   = (double)y_cd * cd2rad;
        
        {
          std::lock_guard<std::mutex> lk(m_);
          phi_meas_      = roll;
          theta_meas_    = pitch;
          last_imu_time_ = this->now();
        }
        
        // Publish in radians
        geometry_msgs::msg::Vector3 v_rad;
        v_rad.x = roll; 
        v_rad.y = pitch; 
        v_rad.z = yaw;
        rpy_pub_->publish(v_rad);
        
        // Publish in degrees
        geometry_msgs::msg::Vector3 v_deg;
        v_deg.x = rad2deg(roll); 
        v_deg.y = rad2deg(pitch); 
        v_deg.z = rad2deg(yaw);
        rpy_deg_pub_->publish(v_deg);
      }
      // Motor roll reply
      else if (id == (uint32_t)(0x240 + motor_roll_id_) && f.can_dlc >= 8) {
        parseMotorReply(f, motor_roll_state_, temp_roll_pub_);
      }
      // Motor pitch reply
      else if (id == (uint32_t)(0x240 + motor_pitch_id_) && f.can_dlc >= 8) {
        parseMotorReply(f, motor_pitch_state_, temp_pitch_pub_);
      }
    }
  }

  void parseMotorReply(const can_frame& f, MotorState& state, 
                       rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr temp_pub) {
    // Protocol section 2.21.3 reply format
    state.temperature_c = (int8_t)f.data[1];
    int16_t iq_raw      = (int16_t)((f.data[3] << 8) | f.data[2]);
    int16_t speed_raw   = (int16_t)((f.data[5] << 8) | f.data[4]);
    int16_t angle_raw   = (int16_t)((f.data[7] << 8) | f.data[6]);
    
    state.current_A  = iq_raw * 0.01;
    state.speed_dps  = speed_raw;
    state.angle_deg  = angle_raw;
    state.last_update = this->now();
    state.valid = true;
    
    // Publish temperature as Float32
    std_msgs::msg::Float32 temp_msg;
    temp_msg.data = state.temperature_c;
    temp_pub->publish(temp_msg);
    
    RCLCPP_DEBUG(get_logger(), 
                 "Motor: T=%d°C I=%.2fA spd=%ddps ang=%d°",
                 state.temperature_c, state.current_A, state.speed_dps, state.angle_deg);
  }

  void controlStep() {
    rclcpp::Time now = this->now();
    
    // Check manual mode timeout
    {
      std::lock_guard<std::mutex> lk(m_);
      if (manual_mode_ && (now - last_manual_time_).seconds() > manual_timeout_s_) {
        manual_mode_ = false;
        RCLCPP_INFO(get_logger(), "Manual mode timeout - switching to automatic control");
      }
    }
    
    double q1_cmd, q2_cmd;
    
    // Check if in manual mode
    bool is_manual;
    {
      std::lock_guard<std::mutex> lk(m_);
      is_manual = manual_mode_;
    }
    
    if (is_manual) {
      // Manual control mode
      std::lock_guard<std::mutex> lk(m_);
      q1_cmd = manual_roll_cmd_rad_;
      q2_cmd = manual_pitch_cmd_rad_;
      
      // Clamp manual commands
      q1_cmd = clamp(q1_cmd, -max_rad_, max_rad_);
      q2_cmd = clamp(q2_cmd, -max_rad_, max_rad_);
      
    } else {
      // Automatic stabilization control
      
      // Check IMU timeout
      {
        std::lock_guard<std::mutex> lk(m_);
        if ((now - last_imu_time_).seconds() > imu_timeout_s_ && 
            last_imu_time_.nanoseconds() > 0) {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                               "IMU timeout! Last update %.2fs ago",
                               (now - last_imu_time_).seconds());
          return;
        }
      }

      double phi, theta;
      {
        std::lock_guard<std::mutex> lk(m_);
        phi   = phi_meas_;
        theta = theta_meas_;
      }

      // Compute dt
      double dt = (last_ctrl_time_.nanoseconds() > 0) ? 
                  (now - last_ctrl_time_).seconds() : (1.0 / rate_hz_);
      dt = clamp(dt, 1e-6, 0.1);
      last_ctrl_time_ = now;

      // Numerical differentiation
      double dphi   = (phi   - phi_prev_)   / dt;
      double dtheta = (theta - theta_prev_) / dt;
      phi_prev_   = phi;
      theta_prev_ = theta;

      // PD control
      double e_phi   = roll_ref_  - phi;
      double e_theta = pitch_ref_ - theta;
      
      double u_phi   = kp_roll_  * e_phi  - kd_roll_  * dphi;
      double u_theta = kp_pitch_ * e_theta - kd_pitch_ * dtheta;

      // Integrate control output
      phi_cmd_   += u_phi   * dt;
      theta_cmd_ += u_theta * dt;
      
      // Clamp integrated commands
      phi_cmd_   = clamp(phi_cmd_,   -max_rad_, max_rad_);
      theta_cmd_ = clamp(theta_cmd_, -max_rad_, max_rad_);

      // Apply slew rate limiting
      double max_step = slew_rad_per_s_ * dt;
      auto slew = [&](double target, double curr) -> double {
        double delta = clamp(target - curr, -max_step, max_step);
        return curr + delta;
      };
      
      phi_cmd_filt_   = slew(phi_cmd_,   phi_cmd_filt_);
      theta_cmd_filt_ = slew(theta_cmd_, theta_cmd_filt_);

      // Convert to motor positions
      q1_cmd = q1_zero_ + sigma_roll_  * phi_cmd_filt_;
      q2_cmd = q2_zero_ + sigma_pitch_ * theta_cmd_filt_;
      
      // Additional safety clamp
      q1_cmd = clamp(q1_cmd, q1_zero_ - max_rad_, q1_zero_ + max_rad_);
      q2_cmd = clamp(q2_cmd, q2_zero_ - max_rad_, q2_zero_ + max_rad_);
    }

    // Publish targets in radians
    std_msgs::msg::Float64MultiArray arr_rad;
    arr_rad.data = { q1_cmd, q2_cmd };
    target_pub_->publish(arr_rad);
    
    // Publish targets in degrees
    std_msgs::msg::Float64MultiArray arr_deg;
    arr_deg.data = { rad2deg(q1_cmd), rad2deg(q2_cmd) };
    target_deg_pub_->publish(arr_deg);

    // Send CAN commands if enabled
    if (enable_can_command_) {
      sendPositionCommand(0x140 + motor_roll_id_,  q1_cmd, max_speed_dps_);
      sendPositionCommand(0x140 + motor_pitch_id_, q2_cmd, max_speed_dps_);
    }
  }

  void sendPositionCommand(uint32_t can_id, double q_rad, uint16_t max_speed_dps) {
    uint8_t frame[8]{};
    
    // Command 0xA4: Absolute Position Closed-Loop Control
    frame[0] = 0xA4;
    frame[1] = 0x00;
    
    // Max speed (DATA[2-3]) - uint16_t, 1dps/LSB
    frame[2] = (uint8_t)(max_speed_dps & 0xFF);
    frame[3] = (uint8_t)((max_speed_dps >> 8) & 0xFF);
    
    // Position (DATA[4-7]) - int32_t, 0.01 degree/LSB (centidegrees)
    int32_t pos_centideg = (int32_t)std::llround(q_rad * 18000.0 / M_PI);
    frame[4] = (uint8_t)(pos_centideg & 0xFF);
    frame[5] = (uint8_t)((pos_centideg >> 8) & 0xFF);
    frame[6] = (uint8_t)((pos_centideg >> 16) & 0xFF);
    frame[7] = (uint8_t)((pos_centideg >> 24) & 0xFF);
    
    tx_.sendFrame(can_id, frame, 8);
  }

  void releaseMotorBrakes() {
    uint8_t brake_cmd[8] = {0x77, 0, 0, 0, 0, 0, 0, 0};
    
    tx_.sendFrame(0x140 + motor_roll_id_,  brake_cmd, 8);
    tx_.sendFrame(0x140 + motor_pitch_id_, brake_cmd, 8);
    
    RCLCPP_INFO(get_logger(), "Motor brakes released (cmd 0x77)");
  }

  // Parameters
  std::string can_iface_;
  int imu_id_;
  int motor_roll_id_;
  int motor_pitch_id_;
  double sigma_roll_, sigma_pitch_;
  double q1_zero_, q2_zero_;
  
  bool enable_control_;
  bool enable_can_command_;
  double rate_hz_;
  double kp_roll_, kd_roll_, kp_pitch_, kd_pitch_;
  double roll_ref_, pitch_ref_;
  double max_rad_;
  double slew_rad_per_s_;
  int max_speed_dps_;
  double imu_timeout_s_;
  bool release_brakes_;
  double manual_timeout_s_;

  // State
  std::mutex m_;
  double phi_meas_{0.0}, theta_meas_{0.0};
  rclcpp::Time last_imu_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_ctrl_time_{0, 0, RCL_ROS_TIME};
  double phi_prev_{0.0}, theta_prev_{0.0};

  // Integrated command & slew-filtered
  double phi_cmd_{0.0}, theta_cmd_{0.0};
  double phi_cmd_filt_{0.0}, theta_cmd_filt_{0.0};
  
  // Manual control state
  bool manual_mode_{false};
  double manual_roll_cmd_rad_{0.0}, manual_pitch_cmd_rad_{0.0};
  rclcpp::Time last_manual_time_{0, 0, RCL_ROS_TIME};

  // Motor states
  MotorState motor_roll_state_;
  MotorState motor_pitch_state_;

  // Infrastructure
  CanRx rx_;
  CanTx tx_;
  rclcpp::TimerBase::SharedPtr timer_;
  
  // Publishers - radians
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr rpy_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr target_pub_;
  
  // Publishers - degrees
  rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr rpy_deg_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr target_deg_pub_;
  
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr temp_roll_pub_;
  rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr temp_pitch_pub_;
  
  // Subscriber for manual commands
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr manual_cmd_sub_;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GimbalControlNode>());
  rclcpp::shutdown();
  return 0;
}