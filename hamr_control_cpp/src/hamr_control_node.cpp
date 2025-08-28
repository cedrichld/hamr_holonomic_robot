// hamr_control_node.cpp
#include "hamr_control_cpp/hamr_control_node.hpp"

HamrControlNode::HamrControlNode() : Node("hamr_control_node") {
    // Constructor implementation
    RCLCPP_INFO(this->get_logger(), "HamrControlNode has been initialized.");
}
