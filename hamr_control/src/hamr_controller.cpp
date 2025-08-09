#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "geometry_msgs/msg/pose_with_covariance.hpp"
#include "tf2_msgs/msg/tf_message.hpp"
#include "hamr_interfaces/msg/live_gains.hpp"
#include <cmath>
#include <chrono>
#include <array>

using namespace std::chrono_literals;

class PIAccumulator {
public:
    PIAccumulator() : sum_(0.0) {}
    void add(double value) { sum_ += value; }
    double get() const { return sum_; }
    void reset() { sum_ = 0.0; }
private:
    double sum_;
};

class HamrControlNode : public rclcpp::Node
{
public:
    HamrControlNode()
        : Node("hamr_controller_node")
    {
        this->declare_parameter("r_wheel", 0.0762);
        this->declare_parameter("a_wheel", 0.149556);
        this->declare_parameter("b_wheel", 0.19682);
        this->declare_parameter("P_x", 0.1);
        this->declare_parameter("I_x", 0.005);
        this->declare_parameter("D_x", 0.001);
        this->declare_parameter("P_y", 0.1);
        this->declare_parameter("I_y", 0.005);
        this->declare_parameter("D_y", 0.001);
        this->declare_parameter("P_yaw", 0.5);
        this->declare_parameter("I_yaw", 0.001);
        this->declare_parameter("D_yaw", 0.001);
        this->declare_parameter("control_rate_hz", 100.0);

        r_wheel_ = this->get_parameter("r_wheel").as_double();
        a_wheel_ = this->get_parameter("a_wheel").as_double();
        b_wheel_ = this->get_parameter("b_wheel").as_double();
        P_x_ = this->get_parameter("P_x").as_double();
        I_x_ = this->get_parameter("I_x").as_double();
        D_x_ = this->get_parameter("D_x").as_double();
        P_y_ = this->get_parameter("P_y").as_double();
        I_y_ = this->get_parameter("I_y").as_double();
        D_y_ = this->get_parameter("D_y").as_double();
        P_yaw_ = this->get_parameter("P_yaw").as_double();
        I_yaw_ = this->get_parameter("I_yaw").as_double();
        D_yaw_ = this->get_parameter("D_yaw").as_double();
        control_rate_hz_ = this->get_parameter("control_rate_hz").as_double();

        left_wheel_vel_pub_ = this->create_publisher<std_msgs::msg::Float64>("/left_wheel/cmd_vel", 1);
        right_wheel_vel_pub_ = this->create_publisher<std_msgs::msg::Float64>("/right_wheel/cmd_vel", 1);
        turret_vel_pub_ = this->create_publisher<std_msgs::msg::Float64>("/turret/cmd_vel", 1);
        gains_pub_ = this->create_publisher<hamr_interfaces::msg::LiveGains>("/live_gains", 10);

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/hamr/odom", 1, std::bind(&HamrControlNode::odom_callback, this, std::placeholders::_1));
        tf_sub_ = this->create_subscription<tf2_msgs::msg::TFMessage>(
            "/tf", 1, std::bind(&HamrControlNode::tf_callback, this, std::placeholders::_1));
        reference_sub_ = this->create_subscription<geometry_msgs::msg::PoseWithCovariance>(
            "/reference_trajectory", 1, std::bind(&HamrControlNode::reference_callback, this, std::placeholders::_1));

        control_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000 / control_rate_hz_)),
            std::bind(&HamrControlNode::control_tick, this));

        pose_base_ = nullptr;
        reference_ = nullptr;
        turret_to_base_orientation_received_ = false;
        last_control_time_ = this->now();
        
        err_x_prev_ = 0.0;
        err_y_prev_ = 0.0;
        err_yaw_prev_ = 0.0;
        d_err_x_filt_ = 0.0;
        d_err_y_filt_ = 0.0;
        d_err_yaw_filt_ = 0.0;
        threshold_x_y_ = 0.01;
        threshold_yaw_ = 0.01;
        yaw_ = 0.0;
    }

private:
    void odom_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
    {
        pose_base_ = msg;
    }

    void tf_callback(const tf2_msgs::msg::TFMessage::SharedPtr msg)
    {
        for (const auto& t : msg->transforms)
        {
            if (t.child_frame_id == "turret_link")
            {
                turret_to_base_orientation_ = t.transform.rotation; 
                turret_to_base_orientation_received_ = true;
                break;
            }
        }
    }

    void reference_callback(const geometry_msgs::msg::PoseWithCovariance::SharedPtr msg)
    {
        reference_ = msg;
    }

    void control_tick()
    {
        auto now = this->now();
        auto dt = (now - last_control_time_).seconds();
        last_control_time_ = now;

        if (pose_base_ != nullptr && reference_ != nullptr && turret_to_base_orientation_received_)
        {
            pid_step(dt);
        }
    }

    void pid_step(double dt)
    {
        double err_x = reference_->pose.position.x - pose_base_->pose.pose.position.x;
        double err_y = reference_->pose.position.y - pose_base_->pose.pose.position.y;

        double yaw_des = quat_to_angle(reference_->pose.orientation);
        double yaw_curr_b_w = quat_to_angle(pose_base_->pose.pose.orientation);
        double yaw_curr_t_b = quat_to_angle(turret_to_base_orientation_);
        double yaw_curr_t_w = yaw_curr_b_w + yaw_curr_t_b;
        double err_yaw = wrap_angle(yaw_des - yaw_curr_t_w);

        yaw_ = yaw_curr_t_w;

        I_x_accum_.add(err_x * dt);
        I_y_accum_.add(err_y * dt);
        I_yaw_accum_.add(err_yaw * dt);

        double desired_x_dot = P_x_ * err_x + I_x_ * I_x_accum_.get() + D_x_ * (err_x - err_x_prev_) / dt;
        double desired_y_dot = P_y_ * err_y + I_y_ * I_y_accum_.get() + D_y_ * (err_y - err_y_prev_) / dt;
        double desired_yaw_dot = P_yaw_ * err_yaw + I_yaw_ * I_yaw_accum_.get() + D_yaw_ * (err_yaw - err_yaw_prev_) / dt;

        err_x_prev_ = err_x;
        err_y_prev_ = err_y;
        err_yaw_prev_ = err_yaw;

        publish_joint_cmd(desired_x_dot, desired_y_dot, desired_yaw_dot);
    }

    void publish_joint_cmd(double desired_x_dot, double desired_y_dot, double desired_yaw_dot)
    {
        std_msgs::msg::Float64 right_wheel_omega, left_wheel_omega, turret_omega;
        std::array<double, 3> omegas = compute_velocities({desired_x_dot, desired_y_dot, desired_yaw_dot});

        right_wheel_omega.data = omegas[0];
        left_wheel_omega.data = omegas[1];
        turret_omega.data = omegas[2];

        right_wheel_vel_pub_->publish(right_wheel_omega);
        left_wheel_vel_pub_->publish(left_wheel_omega);
        turret_vel_pub_->publish(turret_omega);
    }

    std::array<double, 3> compute_velocities(const std::array<double, 3>& desired_velocity)
    {
        double c = std::cos(yaw_);
        double s = std::sin(yaw_);
        
        std::array<std::array<double, 3>, 3> J = {{
            {{r_wheel_/2 * (c + s*b_wheel_/a_wheel_), r_wheel_/2 * (c - s*b_wheel_/a_wheel_), 0.0}},
            {{r_wheel_/2 * (-s + c*b_wheel_/a_wheel_), r_wheel_/2 * (-s - c*b_wheel_/a_wheel_), 0.0}},
            {{r_wheel_/(2*a_wheel_), -r_wheel_/(2*a_wheel_), 1.0}}
        }};
        
        return solve_jacobian(J, desired_velocity);
    }

    std::array<double, 3> solve_jacobian(const std::array<std::array<double, 3>, 3>& J, const std::array<double, 3>& desired_velocity)
    {
        std::array<double, 3> omegas;
        
        omegas[0] = J[0][0] * desired_velocity[0] + J[0][1] * desired_velocity[1] + J[0][2] * desired_velocity[2];
        omegas[1] = J[1][0] * desired_velocity[0] + J[1][1] * desired_velocity[1] + J[1][2] * desired_velocity[2];
        omegas[2] = J[2][0] * desired_velocity[0] + J[2][1] * desired_velocity[1] + J[2][2] * desired_velocity[2];
        
        return omegas;
    }

    double quat_to_angle(const geometry_msgs::msg::Quaternion& q)
    {
        return atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
    }

    double wrap_angle(double a)
    {
        return std::fmod((a + M_PI), (2.0 * M_PI)) - M_PI;
    }

private:
    double r_wheel_, a_wheel_, b_wheel_;
    double P_x_, I_x_, D_x_, P_y_, I_y_, D_y_, P_yaw_, I_yaw_, D_yaw_;
    double control_rate_hz_;
    
    nav_msgs::msg::Odometry::SharedPtr pose_base_;
    geometry_msgs::msg::PoseWithCovariance::SharedPtr reference_;
    geometry_msgs::msg::Quaternion turret_to_base_orientation_;
    bool turret_to_base_orientation_received_;

    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_wheel_vel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr right_wheel_vel_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr turret_vel_pub_;
    rclcpp::Publisher<hamr_interfaces::msg::LiveGains>::SharedPtr gains_pub_;
    
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovariance>::SharedPtr reference_sub_;
    rclcpp::TimerBase::SharedPtr control_timer_;

    rclcpp::Time last_control_time_;
    double err_x_prev_, err_y_prev_, err_yaw_prev_;
    double d_err_x_filt_, d_err_y_filt_, d_err_yaw_filt_;
    PIAccumulator I_x_accum_, I_y_accum_, I_yaw_accum_;
    double threshold_x_y_, threshold_yaw_;
    double yaw_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HamrControlNode>());
    rclcpp::shutdown();
    return 0;
}
