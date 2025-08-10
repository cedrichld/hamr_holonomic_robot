#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono_literals;

enum class State { FORWARD, TURN, DONE };

class HamrSquareNode : public rclcpp::Node
{
public:
    HamrSquareNode()
    : Node("hamr_square_node"),
      wheel_speed_(declare_parameter("wheel_speed_rad_s", 3.0)),
      side_length_(declare_parameter("side_length_m", 5.0)),
      turn_angle_deg_(90.0),
      have_start_(false), state_(State::FORWARD),
      side_count_(0)
    {
        left_pub_  = create_publisher<std_msgs::msg::Float64>("/left_wheel/cmd_vel", 1);
        right_pub_ = create_publisher<std_msgs::msg::Float64>("/right_wheel/cmd_vel", 1);

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/hamr/odom", 10,
            [this](const nav_msgs::msg::Odometry::SharedPtr msg){ this->odom_cb(msg); });

        timer_ = create_wall_timer(100ms, [this]{ this->tick(); });
    }

private:
    void odom_cb(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        double x = msg->pose.pose.position.x;
        double y = msg->pose.pose.position.y;

        // yaw extraction from quaternion
        auto &q = msg->pose.pose.orientation;
        double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
        double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        double yaw = std::atan2(siny_cosp, cosy_cosp);

        if (!have_start_) {
            start_x_ = x;
            start_y_ = y;
            start_yaw_ = yaw;
            have_start_ = true;
            return;
        }

        if (state_ == State::FORWARD) {
            double dist = hypot(x - start_x_, y - start_y_);
            if (dist >= side_length_) {
                // prepare for turn
                state_ = State::TURN;
                start_yaw_ = yaw;
            }
        }
        else if (state_ == State::TURN) {
            double dyaw = normalize_angle(yaw - start_yaw_);
            if (fabs(dyaw) >= deg2rad(turn_angle_deg_)) {
                // done turning
                side_count_++;
                if (side_count_ >= 4) {
                    state_ = State::DONE;
                } else {
                    state_ = State::FORWARD;
                    start_x_ = x;
                    start_y_ = y;
                }
            }
        }
    }

    void tick()
    {
        std_msgs::msg::Float64 left, right;

        if (!have_start_) {
            left.data = right.data = 0.0;
        }
        else if (state_ == State::FORWARD) {
            left.data = wheel_speed_;
            right.data = wheel_speed_;
        }
        else if (state_ == State::TURN) {
            // spin in place: left forward, right backward
            left.data = wheel_speed_;
            right.data = -wheel_speed_;
        }
        else { // DONE
            left.data = right.data = 0.0;
        }

        left_pub_->publish(left);
        right_pub_->publish(right);
    }

    static double normalize_angle(double a) {
        while (a > M_PI) a -= 2.0 * M_PI;
        while (a < -M_PI) a += 2.0 * M_PI;
        return a;
    }

    static double deg2rad(double d) {
        return d * M_PI / 180.0;
    }

    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_pub_, right_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    double wheel_speed_;
    double side_length_;
    double turn_angle_deg_;
    bool have_start_;
    State state_;
    int side_count_;

    double start_x_, start_y_, start_yaw_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(make_shared<HamrSquareNode>());
    rclcpp::shutdown();
    return 0;
}
