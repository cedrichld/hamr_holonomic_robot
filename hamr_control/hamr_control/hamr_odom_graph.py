#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage # to access TFs (for turret relative angle) - could also be used for position esimation with "encoders"

from hamr_interfaces.msg import ReferenceTraj

import math
import matplotlib.pyplot as plt
import time

### - - UTILITIES - - ###
def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def quat_to_angle(q):
    return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

class OdomGraphNode(Node):
    def __init__(self):
        super().__init__("hamr_odom_graph_node")
        self.odom_sub_ = self.create_subscription(
            Odometry, "/hamr/odom", self.odom_callback, 10)
        self.tf_sub_ = self.create_subscription(
            TFMessage, "/tf", self.callback_tf, 1)
        self.reference_sub_ = self.create_subscription(
            ReferenceTraj, "/reference_trajectory", self.callback_reference, 1)
        
        self.get_logger().info("OdomGraphNode started.")
        
        # current values
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw_b_w = 0.0
        self.curr_yaw_t_b = 0.0

        self.reference_x = 0.0
        self.reference_y = 0.0
        self.reference_yaw = 0.0

    def odom_callback(self, msg: Odometry):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_yaw_b_w = quat_to_angle(msg.pose.pose.orientation)
    
    def callback_reference(self, msg: ReferenceTraj):
        self.reference_x = msg.x
        self.reference_y = msg.y
        self.reference_yaw = msg.yaw

    def callback_tf(self, msg: TFMessage):
        ''' Look through all TFs and find turret_link to get it's Quaternion '''
        for t in msg.transforms:
            if t.child_frame_id == "turret_link" and t.header.frame_id  == "base_link":
                self.curr_yaw_t_b = quat_to_angle(t.transform.rotation)
                break

def main(args=None):
    rclpy.init(args=args)
    node = OdomGraphNode()

    ### Fig 1: Odometry
    plt.ion()
    fig1, ax1 = plt.subplots()
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Value')
    ax1.set_title('Odometry: x, y, yaw')
    line_x,   = ax1.plot([], [], label='x', color='blue', linewidth=2)
    line_y,   = ax1.plot([], [], label='y', color='green', linewidth=2)
    line_yaw, = ax1.plot([], [], label='yaw', color='red', linewidth=2)
    line_x_ref,   = ax1.plot([], [], label='x_ref', color='blue', linewidth=1, linestyle='dashed')
    line_y_ref,   = ax1.plot([], [], label='y_ref', color='green', linewidth=1, linestyle='dashed')
    line_yaw_ref, = ax1.plot([], [], label='yaw_ref', color='red', linewidth=1, linestyle='dashed')
    ax1.legend()

    # data buffers
    t_buf, x_buf, y_buf, yaw_buf = [], [], [], []
    x_buf_ref, y_buf_ref, yaw_buf_ref = [], [], []
    t0 = time.time()

    # Limit history
    MAX_N = 300  # 30s at 10 Hz

    def trim(*lists):
        if MAX_N is None:
            return
        for L in lists:
            if len(L) > MAX_N:
                del L[:len(L) - MAX_N]

    try:
        while rclpy.ok():
            # pump ROS callbacks
            rclpy.spin_once(node, timeout_sec=0.1)

            # record timestamp and values
            t = time.time() - t0
            t_buf.append(t)
            x_buf.append(node.curr_x)
            y_buf.append(node.curr_y)
            yaw_buf.append(wrap_angle(node.curr_yaw_b_w + node.curr_yaw_t_b))
            x_buf_ref.append(node.reference_x)
            y_buf_ref.append(node.reference_y)
            yaw_buf_ref.append(wrap_angle(node.reference_yaw))

            # trim history
            trim(t_buf, x_buf, y_buf, yaw_buf, x_buf_ref, y_buf_ref, yaw_buf_ref)

            # update ODOM lines
            line_x.set_data(t_buf, x_buf)
            line_y.set_data(t_buf, y_buf)
            line_yaw.set_data(t_buf, yaw_buf)
            line_x_ref.set_data(t_buf, x_buf_ref)
            line_y_ref.set_data(t_buf, y_buf_ref)
            line_yaw_ref.set_data(t_buf, yaw_buf_ref)
            ax1.relim(); ax1.autoscale_view()

            plt.pause(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    
    
if __name__ == "__main__":
    main()