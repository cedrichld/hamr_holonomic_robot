#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from hamr_interfaces.msg import StateError
from hamr_interfaces.msg import ReferenceTraj

import math
import numpy as np

### PROBLEM:
## This current configuration does not work well most-probably bc the points are way too close
    # to the robot. the robot had way better control when the points were a meter or 2 away.
## TODO:
    # ?Line following with a set distance, following a trajectory (set of points that are more spread out?)
    # ?Maybe more sophisticated trajectory generation (e.g. using splines)

class TrajectoryNode(Node):
    def __init__(self):
        super().__init__("waypoint_traj_node")
        self.reference_timer_hz = self.declare_parameter("reference_timer_hz", 100.).value

        self.state_error_sub_ = self.create_subscription(
            StateError, "/state_error", self.callback_state_error, 1)
        self.reference_trajectory_pub_ = self.create_publisher(
            ReferenceTraj, "/reference_trajectory", 5
        )

        self.last_reference_time = self.get_clock().now()
        self.reference_timer_ = self.create_timer(
            1 / self.reference_timer_hz, self.reference_udpdate)
        
        self.err_xy = math.inf
        self.err_yaw = math.inf

        points = np.array([ # x, y, yaw
            # [0.0, 0.0], # SQUARE
            # [5.0, 0.0],
            # [5.0, 5.0],
            # [0.0, 5.0],
            # [0.0, 0.0],

            [0.0, 0.0, 0.0], # TRIANGLE
            [5.0, 2.5, -1.0],
            [0.0, 5.0, -1.0],
            [0.0, 0.0, 1.0],
        ])
        self.trajectory = WaypointTraj(points)
    
    def callback_state_error(self, msg: StateError):
        self.err_xy = math.hypot(msg.err_x, msg.err_y)
        self.err_yaw = msg.err_yaw
    
    def reference_udpdate(self):
        now = self.get_clock().now()
        t = (now - self.last_reference_time).nanoseconds * 1e-9
        x, y, yaw, x_dot, y_dot, yaw_dot = self.trajectory.update(t)

        pose = ReferenceTraj()
        pose.x, pose.y, pose.yaw, pose.x_dot, pose.y_dot, pose.yaw_dot = float(x), float(y), float(yaw), float(x_dot), float(y_dot), float(yaw_dot)
        self.reference_trajectory_pub_.publish(pose)
        self.get_logger().info("pose: x=%.2f, y=%.2f, yaw=%.2f" % (x, y, yaw))
        if t >= self.trajectory.total_time:
            self.get_logger().info("Resetting traj")
            self.last_reference_time = self.get_clock().now()


class WaypointTraj(object):
    def __init__(self, points, speed=0.5):
        """
        Inputs: points, (N, 2) array of N waypoint coordinates in 2D
        """
        points = np.array(points)

        # Keep points properly shaped
        if points.ndim == 1:
            if points.size % 3 != 0:
                raise ValueError("points.size % 3 != 0")
            points = points.reshape(-1, 3)
        elif points.ndim == 3 and points.shape[1] != 3:
            if points.shape[0] == 3:
                points = points.T
            else:
                raise ValueError("points.shape[0] != 3")

        self.points = points.astype(float)
        self.speed = float(speed)
        self.N = len(points)

        def wrap_angle(a):
            return np.arctan2(np.sin(a), np.cos(a))

        d = np.diff(self.points, axis=0) # (N-1,3)
        d[:, 2] = wrap_angle(d[:, 2]) # wrap yaw deltas

        self.segment_lengths = np.linalg.norm(d, axis=1, keepdims=True) # (N -1, 1)
        self.l_hat = d / (self.segment_lengths + 1e-8) # unit directions
        
        self.segment_times = self.segment_lengths.flatten() / self.speed
        self.t_start = np.hstack(([0.0], np.cumsum(self.segment_times)))
        self.total_time = float(self.t_start[-1])
        

    def update(self, t: float):
        """
        Given the present time, return the desired flat output
        Inputs
            t, time, s
        Outputs
            q, position
            yaw, turret
        """
        if t >= self.total_time:
            x_last, y_last, yaw_last = self.points[-1]
            return float(x_last), float(y_last), float(yaw_last), 0.0, 0.0, 0.0

        seg = int(np.searchsorted(self.t_start, t, side='right') - 1)
        dt = t - self.t_start[seg]

        q = self.points[seg] + self.l_hat[seg] * self.speed * dt # (3,)
        q_dot = self.l_hat[seg] * self.speed # (3,)

        #      x            y            yaw          x_dot            y_dot            yaw_dot
        return float(q[0]), float(q[1]), float(q[2]), float(q_dot[0]), float(q_dot[1]), float(q_dot[2])

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    rclpy.spin(node)
    rclpy.shutdown()
    
    
if __name__ == "__main__":
    main()