#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include <chrono>

using namespace std::chrono_literals;

class HamrStraightNode : public rclcpp::Node
{
public:
    HamrStraightNode()
        : Node("hamr_straight_node")
    {
        // Publishers for wheel velocities
        left_wheel_vel_pub_ = this->create_publisher<std_msgs::msg::Float64>("/left_wheel/cmd_vel", 1);
        right_wheel_vel_pub_ = this->create_publisher<std_msgs::msg::Float64>("/right_wheel/cmd_vel", 1);

        // Run at 10 Hz
        control_timer_ = this->create_wall_timer(100ms, std::bind(&HamrStraightNode::drive_straight, this));
    }

private:
    void drive_straight()
    {
        std_msgs::msg::Float64 left_cmd;
        std_msgs::msg::Float64 right_cmd;

        // Set both wheels to the same velocity (rad/s)
        left_cmd.data = 1.0;   // Adjust this for desired speed
        right_cmd.data = 1.0;

        left_wheel_vel_pub_->publish(left_cmd);
        right_wheel_vel_pub_->publish(right_cmd);

        RCLCPP_INFO(this->get_logger(), "Driving straight: left=%.2f, right=%.2f",
                    left_cmd.data, right_cmd.data);
    }

    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_wheel_vel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr right_wheel_vel_pub_;
    rclcpp::TimerBase::SharedPtr control_timer_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HamrStraightNode>());
    rclcpp::shutdown();
    return 0;
}
