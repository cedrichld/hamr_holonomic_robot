#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>
#include <std_msgs/msg/int32.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <thread>
#include <cerrno>

#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/ioctl.h>

using namespace std::chrono_literals;

namespace {

constexpr uint16_t MAGIC = 0xCAFE;
constexpr uint16_t VER   = 1;
constexpr uint16_t TYPE_CMD3 = 0x0011;
constexpr uint16_t TYPE_ENC = 0x0003; // ESP->PC: encoder ticks (L,R,T)
constexpr uint16_t TYPE_POSE = 0x0004; // ESP->PC: pose (x,y,theta) + covariance

// Wire packet (packed)
#pragma pack(push,1)
struct PacketCmd3 {
  uint16_t magic;     // 0xCAFE
  uint16_t ver;       // 1
  uint16_t type;      // 0x0011
  uint32_t seq;
  uint64_t t_tx_ns;   // host monotonic send time
  float left;
  float right;
  float turret;
  uint16_t crc16;     // CRC32 folded to 16 bits
};
#pragma pack(pop)

static_assert(sizeof(PacketCmd3) == 2+2+2+4+8+4+4+4+2, "Packet size mismatch");

#pragma pack(push,1)
struct PacketEnc {
  uint16_t magic;     // 0xCAFE
  uint16_t ver;       // 1
  uint16_t type;      // 0x0003
  uint32_t seq;
  uint64_t t_tx_ns;   // device side send timestamp (ns)
  int32_t  ticksL;    // left encoder ticks
  int32_t  ticksR;    // right encoder ticks
  int32_t  ticksT;    // turret encoder ticks
  uint16_t crc16;     // CRC32 folded to 16 bits
};
#pragma pack(pop)

static_assert(sizeof(PacketEnc) == 32, "PacketEnc size mismatch");

// Pose Packet
#pragma pack(push,1)
struct PacketPose {
  uint16_t magic;     // 0xCAFE
  uint16_t ver;       // 1
  uint16_t type;      // 0x0004
  uint32_t seq;
  uint64_t t_tx_ns;   // device side send timestamp (ns)
  float x;            // meters
  float y;            // meters
  float theta;        // radians
  float sigma_x, sigma_y, sigma_theta; // Uncertainties (standard deviations)
  uint8_t ekf_status; // 0=odometry_only, 1=ekf_fused, 2=imu_invalid
  uint16_t crc16;    // CRC32 folded to 16 bits
};
#pragma pack(pop)

static_assert(sizeof(PacketPose) == 2+2+2+4+8+4+4+4+4+4+4+1+2, "PacketPose size mismatch");

// Simple CRC32 -> fold to 16 bits 
uint16_t crc16_fold(const uint8_t* data, size_t n) {
  uint32_t c = 0xFFFFFFFFu;
  for (size_t i = 0; i < n; ++i) {
    c ^= data[i];
    for (int k = 0; k < 8; ++k) {
      c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
    }
  }
  c ^= 0xFFFFFFFFu;
  return static_cast<uint16_t>(c & 0xFFFFu);
}

speed_t baud_to_speed_t(int baud) {
  switch (baud) {
    case 9600: return B9600;
    case 19200: return B19200;
    case 38400: return B38400;
    case 57600: return B57600;
    case 115200: return B115200;
#ifdef B230400
    case 230400: return B230400;
#endif
#ifdef B460800
    case 460800: return B460800;
#endif
#ifdef B921600
    case 921600: return B921600;
#endif
    default: return B115200;
  }
}

class SerialPort {
public:
  SerialPort() = default;
  ~SerialPort() { close(); }

  bool open(const std::string& dev, int baud) {
    close();
    fd_ = ::open(dev.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (fd_ < 0) {
      perror("open serial");
      return false;
    }

    struct termios tio{};
    if (tcgetattr(fd_, &tio) != 0) {
      perror("tcgetattr");
      close();
      return false;
    }

    cfmakeraw(&tio);
    tio.c_cflag |= (CLOCAL | CREAD);
    tio.c_cflag &= ~CSTOPB;            // 1 stop
    tio.c_cflag &= ~PARENB;            // no parity
    tio.c_cflag &= ~CSIZE;
    tio.c_cflag |= CS8;                // 8 data bits

    const speed_t spd = baud_to_speed_t(baud);
    cfsetispeed(&tio, spd);
    cfsetospeed(&tio, spd);

    tio.c_cc[VMIN]  = 0;   // non-blocking read
    tio.c_cc[VTIME] = 0;   // no inter-char timer

    if (tcsetattr(fd_, TCSANOW, &tio) != 0) {
      perror("tcsetattr");
      close();
      return false;
    }

    // clear buffers
    tcflush(fd_, TCIOFLUSH);
    return true;
  }

  void close() {
    if (fd_ >= 0) {
      ::close(fd_);
      fd_ = -1;
    }
  }

  bool write_all(const uint8_t* data, size_t size) {
    if (fd_ < 0) return false;
    size_t sent = 0;
    while (sent < size) {
      ssize_t n = ::write(fd_, data + sent, size - sent);
      if (n > 0) {
        sent += static_cast<size_t>(n);
      } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        // Brief sleep to yield
        ::usleep(100);
      } else {
        perror("serial write");
        return false;
      }
    }
    return true;
  }

  int fd() const { return fd_; }

private:
  int fd_ = -1;
};

} // namespace

class RelayNode : public rclcpp::Node {
public:
  RelayNode() : Node("hamr_uros_bridge") {
    // Parameters
    serial_port_ = this->declare_parameter<std::string>("serial_port", "/dev/ttyUSB0");
    baud_        = this->declare_parameter<int>("baud", 460800);
    tx_rate_hz_  = this->declare_parameter<double>("tx_rate_hz", 100.0);
    frame_id_    = this->declare_parameter<std::string>("frame_id", "base_link");
    odom_frame_id_ = this->declare_parameter<std::string>("odom_frame_id", "odom");

    // Open serial
    if (!serial_.open(serial_port_, baud_)) {
      RCLCPP_FATAL(get_logger(), "Failed to open serial %s @ %d", serial_port_.c_str(), baud_);
      throw std::runtime_error("serial open failed");
    }
    RCLCPP_INFO(get_logger(), "Serial open: %s @ %d", serial_port_.c_str(), baud_);

    // Subscriptions
    using std::placeholders::_1;
    sub_left_ = create_subscription<std_msgs::msg::Float64>(
      "/left_wheel/cmd_vel", rclcpp::QoS(1).best_effort(),
      std::bind(&RelayNode::left_cb, this, _1));
    sub_right_ = create_subscription<std_msgs::msg::Float64>(
      "/right_wheel/cmd_vel", rclcpp::QoS(1).best_effort(),
      std::bind(&RelayNode::right_cb, this, _1));
    sub_turret_ = create_subscription<std_msgs::msg::Float64>(
      "/turret/cmd_vel", rclcpp::QoS(1).best_effort(),
      std::bind(&RelayNode::turret_cb, this, _1));

    // Publishers (raw ticks)
    pub_ticks_l_ = create_publisher<std_msgs::msg::Int32>("/left_wheel/encoder_ticks", 10);
    pub_ticks_r_ = create_publisher<std_msgs::msg::Int32>("/right_wheel/encoder_ticks", 10);
    pub_ticks_t_ = create_publisher<std_msgs::msg::Int32>("/turret/encoder_ticks", 10);

    // Publishers for pose data
    pub_pose_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>("/robot_pose", 10);
    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>("/odom", 10);

    // Timer for TX
    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, tx_rate_hz_));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&RelayNode::tx_tick, this));

    // Start RX thread
    running_ = true;
    reader_ = std::thread([this]{ rx_loop(); });
  }

  ~RelayNode() override {
    running_ = false;
    if (reader_.joinable()) reader_.join();
  }

private:
  void left_cb(const std_msgs::msg::Float64 & msg)  { left_ = static_cast<float>(msg.data); }
  void right_cb(const std_msgs::msg::Float64 & msg) { right_ = static_cast<float>(msg.data); }
  void turret_cb(const std_msgs::msg::Float64 & msg){ turret_ = static_cast<float>(msg.data); }

  void tx_tick() {
    PacketCmd3 pkt{};
    pkt.magic   = MAGIC;
    pkt.ver     = VER;
    pkt.type    = TYPE_CMD3;
    pkt.seq     = ++seq_;
    pkt.t_tx_ns = static_cast<uint64_t>(this->now().nanoseconds()); // ROS time; OK for bookkeeping
    pkt.left    = left_.load(std::memory_order_relaxed);
    pkt.right   = right_.load(std::memory_order_relaxed);
    pkt.turret  = turret_.load(std::memory_order_relaxed);

    pkt.crc16 = crc16_fold(reinterpret_cast<const uint8_t*>(&pkt), sizeof(PacketCmd3) - 2);

    const bool ok = serial_.write_all(reinterpret_cast<const uint8_t*>(&pkt), sizeof(pkt));
    if (!ok) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Serial write failed");
    }
  }

  // Helper to convert quaternion from yaw
  void yaw_to_quaternion(float yaw, float& qx, float& qy, float& qz, float& qw) {
    float half_yaw = yaw * 0.5f;
    qx = 0.0f;
    qy = 0.0f;
    qz = std::sin(half_yaw);
    qw = std::cos(half_yaw);
  }
  
  // Publish pose data as ROS message
  void publish_pose_data(const PacketPose& p) {
    auto stamp = rclcpp::Clock(RCL_ROS_TIME).now();
    // PoseWithCovarianceStamped
    auto pose_msg = geometry_msgs::msg::PoseWithCovarianceStamped();
    pose_msg.header.stamp = stamp;
    pose_msg.header.frame_id = odom_frame_id_;

    pose_msg.pose.pose.position.x = p.x;
    pose_msg.pose.pose.position.y = p.y;
    pose_msg.pose.pose.position.z = 0.0f;

    float qx, qy, qz, qw;
    yaw_to_quaternion(p.theta, qx, qy, qz, qw);
    pose_msg.pose.pose.orientation.x = qx;
    pose_msg.pose.pose.orientation.y = qy;
    pose_msg.pose.pose.orientation.z = qz;
    pose_msg.pose.pose.orientation.w = qw;

    // Set 6X6 covariance matrix
    std::fill(pose_msg.pose.covariance.begin(), pose_msg.pose.covariance.end(), 0.0);
    pose_msg.pose.covariance[0] = p.sigma_x * p.sigma_x; // x variance
    pose_msg.pose.covariance[7] = p.sigma_y * p.sigma_y; // y variance
    pose_msg.pose.covariance[35] = p.sigma_theta * p.sigma_theta; // yaw variance

    pub_pose_->publish(pose_msg);
    
    //Publish Odometry
    auto odom_msg = nav_msgs::msg::Odometry();
    odom_msg.header.stamp = stamp;
    odom_msg.header.frame_id = odom_frame_id_;
    odom_msg.child_frame_id = frame_id_;

    odom_msg.pose = pose_msg.pose;

    // Twist covariance
    std::fill(odom_msg.twist.covariance.begin(), odom_msg.twist.covariance.end(), 0.0);
    odom_msg.twist.covariance[0] = 1000.0; // high uncertainty in vx
    odom_msg.twist.covariance[7] = 1000.0; // high uncertainty in vy
    odom_msg.twist.covariance[35] = 1000.0; // high uncertainty in vtheta

    pub_odom_->publish(odom_msg);

    // Log EKF status occasionally
    static auto last_log = this->now();
    if ((stamp - last_log).seconds() > 2.0) {
      const char* status_str = (p.ekf_status == 1) ? "EKF_FUSED" : 
                              (p.ekf_status == 2) ? "IMU_INVALID" : "ODOMETRY_ONLY";
      RCLCPP_INFO(get_logger(), "Pose: (%.3f±%.3f, %.3f±%.3f, %.1f±%.1f°) Status: %s",
                  p.x, p.sigma_x, p.y, p.sigma_y, 
                  p.theta * 180.0f / M_PI, p.sigma_theta * 180.0f / M_PI,
                  status_str);
      last_log = stamp;
    }
  }
  // --- RX loop: parse encoder packets from ESP and publish ---
  void rx_loop() {
    std::vector<uint8_t> buf;
    buf.reserve(512);

    while (rclcpp::ok() && running_) {
      // Non-blocking read from serial
      uint8_t tmp[128];
      ssize_t n = ::read(serial_.fd(), tmp, sizeof(tmp));
      if (n > 0) {
        buf.insert(buf.end(), tmp, tmp + n);
      } else if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
        std::this_thread::sleep_for(1ms);
      } else if (n < 0) {
        // unexpected error: brief backoff
        std::this_thread::sleep_for(5ms);
      }

      // Parse frames
      for (;;) {
        if (buf.size() < 6) break;

        // resync to MAGIC
        uint16_t magic = (uint16_t)buf[0] | ((uint16_t)buf[1] << 8);
        if (magic != MAGIC) { buf.erase(buf.begin()); continue; }

        if (buf.size() < 6) break;
        uint16_t ver  = (uint16_t)buf[2] | ((uint16_t)buf[3] << 8);
        uint16_t type = (uint16_t)buf[4] | ((uint16_t)buf[5] << 8);
        if (ver != VER) { buf.erase(buf.begin()); continue; }

        size_t need = 0;
        if (type == TYPE_ENC) {
          need = sizeof(PacketEnc);
        } else if (type == TYPE_POSE) {
          need = sizeof(PacketPose);
        } else {
          // Unknown type (could add other decoders later) → drop 1 byte
          buf.erase(buf.begin());
          continue;
        }

        if (buf.size() < need) break;

        if (type == TYPE_ENC){
          PacketEnc p{};
          std::memcpy(&p, buf.data(), need);

          uint16_t calc = crc16_fold(reinterpret_cast<const uint8_t*>(&p), need - 2);
          if (calc != p.crc16) {
            // bad frame—drop 1 byte and resync
            buf.erase(buf.begin());
            continue;
          }

        // Good frame: publish ticks
        std_msgs::msg::Int32 mL; mL.data = p.ticksL; pub_ticks_l_->publish(mL);
        std_msgs::msg::Int32 mR; mR.data = p.ticksR; pub_ticks_r_->publish(mR);
        std_msgs::msg::Int32 mT; mT.data = p.ticksT; pub_ticks_t_->publish(mT);
        } else if (type == TYPE_POSE) {
          PacketPose p{};
          std::memcpy(&p, buf.data(), need);

          uint16_t calc = crc16_fold(reinterpret_cast<const uint8_t*>(&p), need - 2);
          if (calc != p.crc16) {
            // bad frame—drop 1 byte and resync
            buf.erase(buf.begin());
            continue;
          }

          // Good pose frame: publish
          publish_pose_data(p);
        }

        // consume this frame
        buf.erase(buf.begin(), buf.begin() + need);
      }
    }
  }

  // Params
  std::string serial_port_;
  int baud_;
  double tx_rate_hz_;
  std::string frame_id_;
  std::string odom_frame_id_;

  // Serial
  SerialPort serial_;

  // State (atomic so callbacks + timer are safe)
  std::atomic<float> left_{0.0f}, right_{0.0f}, turret_{0.0f};
  uint32_t seq_{0};

  // ROS
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sub_left_, sub_right_, sub_turret_;
  rclcpp::Publisher<std_msgs::msg::Int32>::SharedPtr pub_ticks_l_, pub_ticks_r_, pub_ticks_t_;
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_pose_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Reader thread
  std::thread reader_;
  std::atomic<bool> running_{false};
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<RelayNode>());
  } catch (const std::exception& e) {
    fprintf(stderr, "RelayNode exception: %s\n", e.what());
  }
  rclcpp::shutdown();
  return 0;
}
