#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/float64.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "nav_msgs/msg/path.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace std;
using namespace std::chrono_literals;

enum class State { WAITING_FOR_PATH, FOLLOWING_PATH, GOAL_REACHED, STOPPED };

struct Point {
    double x, y;
    Point(double x = 0, double y = 0) : x(x), y(y) {}
    
    double distance_to(const Point& other) const {
        return std::hypot(x - other.x, y - other.y);
    }
};

static double normalize_angle(double a) {
    while (a > M_PI) a -= 2.0 * M_PI;
    while (a <= -M_PI) a += 2.0 * M_PI;
    return a;
}

static double deg2rad(double d) { return d * M_PI / 180.0; }

class HamrPathFollower : public rclcpp::Node {
public:
    HamrPathFollower()
    : Node("hamr_path_follower"),
      max_linear_speed_(declare_parameter("max_linear_speed", 2.0)),      
      max_angular_speed_(declare_parameter("max_angular_speed", 4.0)),    
      wheel_base_(declare_parameter("wheel_base_m", 0.16)),               
      lookahead_distance_(declare_parameter("lookahead_distance_m", 0.8)),
      position_tolerance_(declare_parameter("position_tolerance_m", 0.3)),
      angle_tolerance_(deg2rad(declare_parameter("angle_tolerance_deg", 10.0))),
      k_linear_(declare_parameter("k_linear", 1.5)),                      
      k_angular_(declare_parameter("k_angular", 3.0)),
      goal_timeout_s_(declare_parameter("goal_timeout_s", 2.0)),
      use_pure_pursuit_(declare_parameter("use_pure_pursuit", true))
    {
        left_pub_ = create_publisher<std_msgs::msg::Float64>("/left_wheel/cmd_vel", 1);
        right_pub_ = create_publisher<std_msgs::msg::Float64>("/right_wheel/cmd_vel", 1);
        cmd_vel_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 1);

        odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
            "/hamr/odom", 40,
            [this](const nav_msgs::msg::Odometry::SharedPtr m){ this->onOdometry(m); });

        path_sub_ = create_subscription<nav_msgs::msg::Path>(
            "/astar/path", 10,
            [this](const nav_msgs::msg::Path::SharedPtr p){ this->onPath(p); });

        timer_ = create_wall_timer(20ms, [this]{ this->tick(); });

        state_ = State::WAITING_FOR_PATH;
        RCLCPP_INFO(this->get_logger(), "HAMR Path Follower initialized. Waiting for path...");
    }

private:
    void onOdometry(const nav_msgs::msg::Odometry::SharedPtr &msg) {
        curr_x_ = msg->pose.pose.position.x;
        curr_y_ = msg->pose.pose.position.y;

        const auto &q = msg->pose.pose.orientation;
        const double s = 2.0 * (q.w * q.z + q.x * q.y);
        const double c = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
        curr_yaw_ = std::atan2(s, c);

        have_odom_ = true;
    }

    void onPath(const nav_msgs::msg::Path::SharedPtr &msg) {
        if (msg->poses.empty()) {
            RCLCPP_WARN(this->get_logger(), "Received empty path");
            return;
        }

        path_waypoints_.clear();
        for (const auto& pose : msg->poses) {
            path_waypoints_.emplace_back(pose.pose.position.x, pose.pose.position.y);
        }

        current_waypoint_idx_ = 0;
        state_ = State::FOLLOWING_PATH;
        goal_timer_ = this->now();

        RCLCPP_INFO(this->get_logger(), "Received new path with %zu waypoints", path_waypoints_.size());
        RCLCPP_INFO(this->get_logger(), "Goal: (%.2f, %.2f)", 
                   path_waypoints_.back().x, path_waypoints_.back().y);
    }

    void tick() {
        if (!have_odom_) {
            publishZero();
            return;
        }

        switch (state_) {
            case State::WAITING_FOR_PATH:
                publishZero();
                break;
            case State::FOLLOWING_PATH:
                followPath();
                break;
            case State::GOAL_REACHED:
                publishZero();
                checkForNewPath();
                break;
            case State::STOPPED:
                publishZero();
                break;
        }
    }

    void followPath() {
        if (path_waypoints_.empty()) {
            state_ = State::WAITING_FOR_PATH;
            return;
        }

        Point current_pos(curr_x_, curr_y_);
        Point goal = path_waypoints_.back();
        
        // Check if we've reached the final goal
        double goal_distance = current_pos.distance_to(goal);
        if (goal_distance < position_tolerance_) {
            state_ = State::GOAL_REACHED;
            RCLCPP_INFO(this->get_logger(), "Goal reached! Distance: %.3f", goal_distance);
            return;
        }

        // Check for timeout
        if ((this->now() - goal_timer_).seconds() > goal_timeout_s_ * path_waypoints_.size()) {
            RCLCPP_WARN(this->get_logger(), "Path following timeout");
            state_ = State::STOPPED;
            return;
        }

        Point target_point;
        if (use_pure_pursuit_) {
            target_point = findLookaheadPoint();
        } else {
            target_point = findNextWaypoint();
        }

        // Calculate control commands
        double dx = target_point.x - curr_x_;
        double dy = target_point.y - curr_y_;
        double distance = std::hypot(dx, dy);
        
        if (distance < 0.05) {
            publishZero();
            return;
        }

        // Calculate desired heading
        double desired_heading = std::atan2(dy, dx);
        double heading_error = normalize_angle(desired_heading - curr_yaw_);
        
        // Speed control based on heading error and distance to goal
        double speed_factor = 1.0;
        if (std::fabs(heading_error) > deg2rad(30.0)) {
            speed_factor *= 0.5; // Slow down for sharp turns
        }
        if (goal_distance < 1.0) {
            speed_factor *= std::max(0.3, goal_distance); // Slow down near goal
        }

        double linear_vel = std::min(max_linear_speed_ * speed_factor, k_linear_ * distance);
        double angular_vel = k_angular_ * heading_error;
        
        // Clamp velocities
        linear_vel = std::clamp(linear_vel, 0.0, max_linear_speed_);
        angular_vel = std::clamp(angular_vel, -max_angular_speed_, max_angular_speed_);

        // Apply minimum speeds to overcome friction
        if (linear_vel > 0.0 && linear_vel < 0.1) linear_vel = 0.1;
        if (std::fabs(angular_vel) > 0.0 && std::fabs(angular_vel) < 0.2) {
            angular_vel = std::copysign(0.2, angular_vel);
        }

        convertToWheelSpeeds(linear_vel, angular_vel);
        publishCmdVel(linear_vel, angular_vel);
    }

    Point findLookaheadPoint() {
        Point current_pos(curr_x_, curr_y_);
        
        // Start from current waypoint and look ahead
        for (size_t i = current_waypoint_idx_; i < path_waypoints_.size(); ++i) {
            double dist = current_pos.distance_to(path_waypoints_[i]);
            
            if (dist >= lookahead_distance_) {
                current_waypoint_idx_ = std::max(current_waypoint_idx_, i > 0 ? i - 1 : 0);
                return path_waypoints_[i];
            }
        }
        
        // If no point is far enough, return the last waypoint
        return path_waypoints_.back();
    }

    Point findNextWaypoint() {
        Point current_pos(curr_x_, curr_y_);
        
        // Advance waypoint if we're close to the current one
        while (current_waypoint_idx_ < path_waypoints_.size() - 1) {
            double dist = current_pos.distance_to(path_waypoints_[current_waypoint_idx_]);
            if (dist < lookahead_distance_ * 0.5) {
                current_waypoint_idx_++;
            } else {
                break;
            }
        }
        
        return path_waypoints_[current_waypoint_idx_];
    }

    void checkForNewPath() {
        // Stay in GOAL_REACHED state until a new path arrives
        // This prevents the robot from moving if the goal is republished
        static auto last_log_time = this->now();
        if ((this->now() - last_log_time).seconds() > 5.0) {
            RCLCPP_INFO(this->get_logger(), "At goal, waiting for new path...");
            last_log_time = this->now();
        }
    }

    void convertToWheelSpeeds(double linear_vel, double angular_vel) {
        double wheel_vel_left = linear_vel - (angular_vel * wheel_base_) / 2.0;
        double wheel_vel_right = linear_vel + (angular_vel * wheel_base_) / 2.0;

        std_msgs::msg::Float64 L, R;
        L.data = wheel_vel_left;
        R.data = wheel_vel_right;
        
        left_pub_->publish(L);
        right_pub_->publish(R);
    }

    void publishCmdVel(double linear, double angular) {
        geometry_msgs::msg::Twist cmd;
        cmd.linear.x = linear;
        cmd.angular.z = angular;
        cmd_vel_pub_->publish(cmd);
    }

    void publishZero() {
        std_msgs::msg::Float64 L, R;
        L.data = R.data = 0.0;
        left_pub_->publish(L);
        right_pub_->publish(R);
        
        geometry_msgs::msg::Twist cmd;
        cmd_vel_pub_->publish(cmd);
    }

    // ROS I/O
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr left_pub_, right_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // Parameters
    double max_linear_speed_;
    double max_angular_speed_;
    double wheel_base_;
    double lookahead_distance_;
    double position_tolerance_;
    double angle_tolerance_;
    double k_linear_;
    double k_angular_;
    double goal_timeout_s_;
    bool use_pure_pursuit_;

    // State
    State state_;
    bool have_odom_{false};
    std::vector<Point> path_waypoints_;
    size_t current_waypoint_idx_{0};
    rclcpp::Time goal_timer_;

    // Current pose
    double curr_x_{0.0}, curr_y_{0.0}, curr_yaw_{0.0};
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HamrPathFollower>());
    rclcpp::shutdown();
    return 0;
}