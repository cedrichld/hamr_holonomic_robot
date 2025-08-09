#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from hamr_interfaces.msg import StateError
from geometry_msgs.msg import PoseWithCovariance, Quaternion

import math
import numpy as np

### PROBLEM:
## This current configuration does not work well most-probably bc the points are way too close
    # to the robot. the robot had way better control when the points were a meter or 2 away.
## TODO:
    # Line following with a set distance, following a trajectory (set of points that are more spread out)
    # TODO: put hz back to where it should be (high) and improve waypoint traj to line following

def yaw_to_quaternion(yaw_rad: float) -> Quaternion:
    """ Roll=Pitch=0, yaw about +Z (ROS REP-103). """
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(0.5 * yaw_rad)
    q.w = math.cos(0.5 * yaw_rad)
    return q

class TrajectoryNode(Node):
    def __init__(self):
        super().__init__("waypoint_traj_node")
        self.reference_timer_hz = self.declare_parameter("reference_timer_hz", 10.).value

        self.state_error_sub_ = self.create_subscription(
            StateError, "/state_error", self.callback_state_error, 1)
        self.reference_trajectory_pub_ = self.create_publisher(
            PoseWithCovariance, "/reference_trajectory", 5
        )

        self.last_reference_time = self.get_clock().now()
        self.reference_timer_ = self.create_timer(
            1 / self.reference_timer_hz, self.reference_udpdate)
        
        self.err_xy = math.inf
        self.err_yaw = math.inf

        points = np.array([
            [0.0, 0.0],
            [5.0, 0.0],
            [5.0, 5.0],
            [0.0, 5.0],
            [0.0, 0.0],
        ])
        self.trajectory = WaypointTraj(points)
    
    def callback_state_error(self, msg: StateError):
        self.err_xy = math.hypot(msg.err_x, msg.err_y)
        self.err_yaw = msg.err_yaw
    
    def reference_udpdate(self):
        now = self.get_clock().now()
        t = (now - self.last_reference_time).nanoseconds * 1e-9
        x, y, yaw = self.trajectory.update(t)

        pose = PoseWithCovariance()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation = yaw_to_quaternion(yaw)
        self.reference_trajectory_pub_.publish(pose)
        self.get_logger().info("pose: x=%d, y=%d, yaw=%d" % (x, y, yaw))


class WaypointTraj(object):
    def __init__(self, points, speed=0.25):
        """
        Inputs: points, (N, 2) array of N waypoint coordinates in 2D
        """
        points = np.array(points)

        # Keep points properly shaped
        if points.ndim == 1:
            if points.size % 2 != 0:
                raise ValueError("points.size % 2 != 0")
            points = points.reshape(-1, 2)
        elif points.ndim == 2 and points.shape[1] != 2:
            if points.shape[0] == 2:
                points = points.T
            else:
                raise ValueError("points.shape[0] != 2")

        self.points = points
        self.speed = speed
        self.N = len(points)
        
        self.l_hat = np.diff(self.points, axis=0)
        self.segment_lengths = np.linalg.norm(self.l_hat, axis=1, keepdims=True)
        self.l_hat = self.l_hat / (self.segment_lengths + 1e-8)
        
        self.segment_times = self.segment_lengths.flatten() / self.speed
        self.t_start = np.hstack(([0], np.cumsum(self.segment_times)))
        self.total_time = float(self.t_start[-1])
        

    def update(self, t):
        """
        Given the present time, return the desired flat output
        Inputs
            t, time, s
        Outputs
            q, position
            yaw, turret
        """
        if t >= self.total_time:
            x_last, y_last = self.points[-1]
            yaw_last = 0.0
            return float(x_last), float(y_last), float(yaw_last)

            
        segment_idx = np.searchsorted(self.t_start, t, side='right') - 1
        delta_t = t - self.t_start[segment_idx]

        q = self.points[segment_idx] + self.l_hat[segment_idx] * self.speed * delta_t
        yaw = 0.0

        return float(q[0]), float(q[1]), float(yaw)
    
def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    rclpy.spin(node)
    rclpy.shutdown()
    
    
if __name__ == "__main__":
    main()