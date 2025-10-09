#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

class OdomComparisonPlotter(Node):
    def __init__(self):
        super().__init__('odom_comparison_plotter')
        
        self.declare_parameter('odom_topic', '/robot_pose')
        self.declare_parameter('vicon_topic', '/HAMR_Base/pose')
        self.declare_parameter('max_points', 1000)
        
        odom_topic = self.get_parameter('odom_topic').value
        vicon_topic = self.get_parameter('vicon_topic').value
        max_pts = self.get_parameter('max_points').value
        
        self.odom_x = deque(maxlen=max_pts)
        self.odom_y = deque(maxlen=max_pts)
        self.vicon_x = deque(maxlen=max_pts)
        self.vicon_y = deque(maxlen=max_pts)
        
        self.odom_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            odom_topic,
            self.odom_callback,
            10
        )
        
        self.vicon_sub = self.create_subscription(
            PoseStamped,
            vicon_topic,
            self.vicon_callback,
            10
        )
        
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.line_odom, = self.ax.plot([], [], 'b-o', label='Odometry', markersize=2)
        self.line_vicon, = self.ax.plot([], [], 'r-o', label='Vicon (Ground Truth)', markersize=2)
        
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.set_title('Odometry vs Vicon Comparison')
        self.ax.legend()
        self.ax.grid(True)
        self.ax.axis('equal')
        
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot, interval=100, blit=False
        )
        
        self.get_logger().info(f'Plotting {odom_topic} vs {vicon_topic}')
    
    def odom_callback(self, msg):
        self.odom_x.append(msg.pose.pose.position.x)
        self.odom_y.append(msg.pose.pose.position.y)
    
    def vicon_callback(self, msg):
        self.vicon_x.append(msg.pose.position.x)
        self.vicon_y.append(msg.pose.position.y)
    
    def update_plot(self, frame):
        if len(self.odom_x) > 0:
            self.line_odom.set_data(self.odom_x, self.odom_y)
        
        if len(self.vicon_x) > 0:
            self.line_vicon.set_data(self.vicon_x, self.vicon_y)
        
        if len(self.odom_x) > 0 or len(self.vicon_x) > 0:
            all_x = list(self.odom_x) + list(self.vicon_x)
            all_y = list(self.odom_y) + list(self.vicon_y)
            
            if all_x and all_y:
                margin = 0.5
                self.ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
                self.ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        
        return self.line_odom, self.line_vicon

def main(args=None):
    rclpy.init(args=args)
    node = OdomComparisonPlotter()
    
    try:
        plt.show()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()