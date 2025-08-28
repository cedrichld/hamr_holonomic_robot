#ifndef HAMR_CONTROL_NODE_HPP
#define HAMR_CONTROL_NODE_HPP

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_with_covariance.hpp" 

// Class definition for HamrControlNode
class HamrControlNode : public rclcpp::Node {
public:
    // Constructor
    HamrControlNode();
};

#endif  // HAMR_CONTROL_NODE_HPP
