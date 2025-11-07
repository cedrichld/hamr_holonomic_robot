#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int16_multi_array.hpp>
#include <linux/can.h>
#include <linux/can/raw.h>
#include <net/if.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstring>

class MotorCanNode : public rclcpp::Node {
public:
  MotorCanNode() : Node("motor_can_node") {
    declare_parameter<std::string>("can_interface", "can0");
    get_parameter("can_interface", can_if_);
    if (openSocket() != 0) {
      RCLCPP_FATAL(get_logger(), "Failed to open CAN socket on %s", can_if_.c_str());
      rclcpp::shutdown();
      return;
    }
    sub_ = create_subscription<std_msgs::msg::UInt16MultiArray>(
      "/motor_command", 10,
      [this](const std_msgs::msg::UInt16MultiArray::SharedPtr msg) {
        // Example: pack first 2 bytes as a throttle command to a single CAN ID.
        // TODO: adapt ID/packing for your motor protocol.
        if (msg->data.empty()) return;
        uint16_t cmd = msg->data[0];
        struct can_frame frame{};
        frame.can_id  = 0x200;           // <-- set your motor CAN ID here
        frame.can_dlc = 2;
        frame.data[0] = static_cast<uint8_t>(cmd & 0xFF);
        frame.data[1] = static_cast<uint8_t>((cmd >> 8) & 0xFF);

        ssize_t n = write(sock_, &frame, sizeof(frame));
        if (n != sizeof(frame)) {
          RCLCPP_WARN(get_logger(), "CAN write failed");
        }
      }
    );
    RCLCPP_INFO(get_logger(), "motor_can_node ready on %s", can_if_.c_str());
  }

  ~MotorCanNode() override { closeSocket(); }

private:
  int openSocket() {
    RCLCPP_INFO(get_logger(), "Opening CAN socket on %s...", can_if_.c_str());
    sock_ = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (sock_ < 0) { perror("socket"); return 1; }

    struct ifreq ifr {};
    std::strncpy(ifr.ifr_name, can_if_.c_str(), IFNAMSIZ - 1);
    if (ioctl(sock_, SIOCGIFINDEX, &ifr) < 0) { perror("ioctl"); return 1; }

    struct sockaddr_can addr {};
    addr.can_family  = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;

    if (bind(sock_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) < 0) {
      perror("bind"); return 1;
    }
    RCLCPP_INFO(get_logger(), "CAN socket opened");
    return 0;
  }

  void closeSocket() {
    if (sock_ >= 0) {
      RCLCPP_INFO(get_logger(), "Closing CAN socket");
      close(sock_);
      sock_ = -1;
    }
  }

  std::string can_if_;
  int sock_{-1};
  rclcpp::Subscription<std_msgs::msg::UInt16MultiArray>::SharedPtr sub_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MotorCanNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
