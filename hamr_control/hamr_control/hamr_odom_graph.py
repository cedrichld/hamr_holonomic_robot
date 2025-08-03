import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

import numpy as np
import matplotlib.pyplot as plt
import time

class OdomGraphNode(Node):
    def __init__(self):
        super().__init__("hamd_odom_graph_node")
        self.odom_sub_ = self.create_subscription(Odometry, "/hamr/odom", self.odom_callback, 10)
        self.get_logger().info("OdomGraphNode started.")
        
        # current values
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_yaw = 0.0

    def odom_callback(self, msg: Odometry):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        self.curr_yaw = np.arctan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

def main(args=None):
    rclpy.init(args=args)
    node = OdomGraphNode()

    # —— Set up live plot ——  
    plt.ion()  
    fig, ax = plt.subplots()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Value')
    ax.set_title('Odometry: x, y, yaw')
    line_x,   = ax.plot([], [], label='x')
    line_y,   = ax.plot([], [], label='y')
    line_yaw, = ax.plot([], [], label='yaw')
    ax.legend()

    # data buffers
    t_buf, x_buf, y_buf, yaw_buf = [], [], [], []
    t0 = time.time()

    try:
        while rclpy.ok():
            # **pump** ROS callbacks
            rclpy.spin_once(node, timeout_sec=0.1)

            # record timestamp and values
            t = time.time() - t0
            t_buf.append(t)
            x_buf.append(node.curr_x)
            y_buf.append(node.curr_y)
            yaw_buf.append(node.curr_yaw)

            # **update** lines
            line_x.set_data(t_buf, x_buf)
            line_y.set_data(t_buf, y_buf)
            line_yaw.set_data(t_buf, yaw_buf)

            # **autoscale** axes
            ax.relim()
            ax.autoscale_view()

            plt.draw()
            plt.pause(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    
    
if __name__ == "__main__":
    main()