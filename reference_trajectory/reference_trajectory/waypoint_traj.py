#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from hamr_interfaces.msg import StateError
from hamr_interfaces.msg import ReferenceTraj

from nav_msgs.msg import Path

import math
import numpy as np

### PROBLEM:
## This current configuration does not work well most-probably bc the points are way too close
    # to the robot. the robot had way better control when the points were a meter or 2 away.
## TODO:
    # Implement server call when new waypoint -> reset I terms
    # SPLINES or more sophisticated trajectory generation
    # More spread out waypoints in turns (based on curvature) > Display waypoints and traj on rviz using marker or smth

def quat_to_angle(q):
    return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

class TrajectoryNode(Node):
    def __init__(self):
        super().__init__("waypoint_traj_node")
        self.v_lin = self.declare_parameter("v_lin", 0.3).value # 0.3 m/s is the max our current HW can do
        self.w_yaw = self.declare_parameter("w_yaw", 1.0).value # 1.0 rad/s

        self.reference_timer_hz = self.declare_parameter("reference_timer_hz", 100).value

        self.reference_trajectory_pub_ = self.create_publisher(
            ReferenceTraj, "/reference_trajectory", 1
        )

        self.last_reference_time = self.get_clock().now()
        self.reference_timer_ = self.create_timer(
            1 / self.reference_timer_hz, self.reference_udpdate)
        self.reference_timer_.cancel()

        self.points_goal_sub_ = self.create_subscription(
            Path, "/astar/path", self.callback_points_goal, 1)

        self.err_xy = math.inf
        self.err_yaw = math.inf

        self.astar_points = None

    def callback_points_goal(self, msg: Path):
        self.reference_timer_.cancel()
        pts = []
        for ps in msg.poses:
            x = ps.pose.position.x
            y = ps.pose.position.y
            yaw = quat_to_angle(ps.pose.orientation)
            pts.append([x, y, yaw])
            self.get_logger().info("Received path point: x=%.2f, y=%.2f, yaw=%.2f" % (x, y, yaw))
            # self.get_logger().info("Orientation: w=%.2f, x=%.2f, y=%.2f, z=%.2f" % \
            #                        (ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z))

        if len(pts) < 2:
            self.get_logger().warn("Received <2 path points; ignoring.")
            return

        self.astar_points = np.asarray(pts, dtype=float)
        # self.reference_timer_.cancel()
        self.trajectory = WaypointTraj(self.astar_points, 
                                       v_lin=self.v_lin, w_yaw=self.w_yaw)
        self.last_reference_time = self.get_clock().now()
        self.reference_timer_ = self.create_timer(
            1 / self.reference_timer_hz, self.reference_udpdate)

    def reference_udpdate(self):
        now = self.get_clock().now()
        t = (now - self.last_reference_time).nanoseconds * 1e-9
        x, y, yaw, x_dot, y_dot, yaw_dot = self.trajectory.update(t)

        pose = ReferenceTraj()
        pose.x, pose.y, pose.yaw, pose.x_dot, pose.y_dot, pose.yaw_dot = float(x), float(y), float(yaw), float(x_dot), float(y_dot), float(yaw_dot)
        self.reference_trajectory_pub_.publish(pose)
        # self.get_logger().info("pose: x=%.2f, y=%.2f, yaw=%.2f" % (x, y, yaw))
        # if t >= self.trajectory.total_time:
        #     self.get_logger().info("Resetting traj")
        #     self.last_reference_time = self.get_clock().now()

    # Used if we want to change parameter during runtime
    def parameters_callback(self, params: list[Parameter]): 
        for p in params:
            if p.name == "v_lin":
                self.trajectory.v_lin = p.value
                self.v_lin = p.value
                self.get_logger().info(f"{p.name} changed to {p.value}")
            elif p.name == "w_yaw":
                self.trajectory.w_yaw = p.value
                self.w_yaw = p.value
                self.get_logger().info(f"{p.name} changed to {p.value}")
            elif p.name == "reference_timer_hz":
                self.reference_timer_hz = p.value
                self.reference_timer_.cancel()
                self.reference_timer_ = self.create_timer(
                    1 / self.reference_timer_hz, self.reference_udpdate)
                self.get_logger().info(f"{p.name} changed to {p.value}")
            
class WaypointTraj(object):
    def __init__(self, points, v_lin=0.6, w_yaw=0.3):
        """
        Inputs: points, (N, 3) array of N waypoint coordinates in 2D with yaw
        """
        points = np.array(points, dtype=float)

        # Keep points properly shaped
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 2:
            raise ValueError("points must be (N>=2, 3) array of [x,y,yaw].")
        
        self.points = points
        self.v_lin = float(v_lin)
        self.w_yaw = float(w_yaw)
        self.N = len(points)

        def wrap_angle(a): return np.arctan2(np.sin(a), np.cos(a))

        d = np.diff(self.points, axis=0) # (N-1, 3)
        d_xy = d[:, :2] # (N-1,2)
        d_yaw = wrap_angle(d[:, 2]) # (N-1,)

        # Durations with separate linear/yaw limits
        eps = 1e-9
        d_xy_norm = np.linalg.norm(d_xy, axis=1) # (N-1,)
        T_lin = d_xy_norm / max(self.v_lin, eps)
        T_yaw = np.abs(d_yaw) / max(self.w_yaw, eps)
        T = np.maximum(T_lin, T_yaw)
        T[T < eps] = eps # avoid zero-length segments

        # Precompute per-segment constant velocities
        self.v_xy = (d_xy / T[:, None]) # (N-1, 2)
        self.w = (d_yaw / T) # (N-1,)

        # Timing
        self.t_start = np.hstack(([0.0], np.cumsum(T))) # (N,)
        self.total_time = float(self.t_start[-1])
        self.last_seg = 0
        

    def update(self, t: float):
        """
        Given the present time, return the desired flat output
        Inputs
            t, time, s
        Outputs
            q, position
            yaw, turret
        """
        def wrap_angle(a):
            return np.arctan2(np.sin(a), np.cos(a))
    
        if t >= self.total_time:
            x_last, y_last, yaw_last = self.points[-1]
            return float(x_last), float(y_last), float(yaw_last), 0.0, 0.0, 0.0

        seg = int(np.searchsorted(self.t_start, t, side='right') - 1)
        if seg > self.last_seg:
            self.last_seg = seg
        dt = t - self.t_start[seg]

        # Clamp dt inside segment just in case of numerical edge
        seg_end = self.t_start[seg + 1]
        if dt < 0.0: dt = 0.0
        if dt > (seg_end - self.t_start[seg]): dt = seg_end - self.t_start[seg]

        # Integrate with constant per-segment velocities
        x0, y0, yaw0 = self.points[seg]
        vx, vy = self.v_xy[seg]
        wyaw = self.w[seg]

        x = x0 + vx * dt
        y = y0 + vy * dt
        yaw = wrap_angle(yaw0 + wyaw * dt)

        #      x         y         yaw         x_dot      y_dot      yaw_dot
        return float(x), float(y), float(yaw), float(vx), float(vy), float(wyaw)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryNode()
    rclpy.spin(node)
    rclpy.shutdown()
    
    
if __name__ == "__main__":
    main()