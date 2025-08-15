#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <string>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#include <sys/ioctl.h>

using namespace std::chrono_literals;

namespace {

constexpr uint16_t MAGIC = 0xCAFE;
constexpr uint16_t VER   = 1;
constexpr uint16_t TYPE_CMD3 = 0x0011;

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

    // Timer for TX
    const auto period = std::chrono::duration<double>(1.0 / std::max(1.0, tx_rate_hz_));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&RelayNode::tx_tick, this));
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

  // Params
  std::string serial_port_;
  int baud_;
  double tx_rate_hz_;

  // Serial
  SerialPort serial_;

  // State (atomic so callbacks + timer are safe)
  std::atomic<float> left_{0.0f}, right_{0.0f}, turret_{0.0f};
  uint32_t seq_{0};

  // ROS
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr sub_left_, sub_right_, sub_turret_;
  rclcpp::TimerBase::SharedPtr timer_;
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
