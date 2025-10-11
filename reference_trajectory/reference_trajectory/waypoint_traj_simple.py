#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from hamr_interfaces.msg import StateError
from hamr_interfaces.msg import ReferenceTraj
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy

import math
import numpy as np

### PROBLEM:
## This current configuration does not work well most-probably bc the points are way too close
    # to the robot. the robot had way better control when the points were a meter or 2 away.
## TODO:
    # Implement server call when new waypoint -> reset I terms
    # SPLINES or more sophisticated trajectory generation
    # More spread out waypoints in turns (based on curvature) > Display waypoints and traj on rviz using marker or smth

class TrajectoryNode(Node):
    def __init__(self):
        super().__init__("waypoint_traj_simple_node")
        v_lin = self.declare_parameter("v_lin", 0.2).value
        w_yaw = self.declare_parameter("w_yaw", 0.5).value

        self.reference_timer_hz = self.declare_parameter("reference_timer_hz", 100).value

        self.state_error_sub_ = self.create_subscription(
            StateError, "/state_error", self.callback_state_error, 1)
        self.reference_trajectory_pub_ = self.create_publisher(
            ReferenceTraj, "/reference_trajectory", 1
        )

        qos_waypoints = QoSProfile(
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            reliability=QoSReliabilityPolicy.RELIABLE
        )
        self.waypoints_path_pub_ = self.create_publisher(
            Path, "/waypoints_path", qos_profile=qos_waypoints
        )

        self.begun = False
        self.last_reference_time = None
        
        self.reference_timer_ = self.create_timer(
            1 / self.reference_timer_hz, self.reference_update)
        
        self.err_xy = math.inf
        self.err_yaw = math.inf

        max_point = 5.0
        origin = 0.0

        def generate_ccw_circle_points(radius=5.0, steps_between=10):
            cx = 0.0
            cy = 0.0 + radius

            # Angles for waypoints (rad)
            # waypoints = [-np.pi/2, -np.pi, -3*np.pi/2, -2*np.pi, -5*np.pi/2] # CW
            waypoints = [-5*np.pi/2, -2*np.pi, -3*np.pi/2, -np.pi, -np.pi/2] # CCW
            pts = []

            # First point explicitly at (0,0,0)
            pts.append([cx, cy - radius, 0.0])

            # Generate ccw points
            for i in range(len(waypoints) - 1):
                th_start = waypoints[i]
                th_end   = waypoints[i + 1]

                # steps_between points between waypoints
                thetas = np.linspace(th_start, th_end, steps_between + 1, endpoint=False)[1:] if i == 0 else \
                        np.linspace(th_start, th_end, steps_between + 1, endpoint=False)

                for th in thetas:
                    x = cx + radius * np.cos(th)
                    y = cy + radius * np.sin(th)
                    pts.append([float(x), float(y), 0.0])

            # Close the loop back to start
            pts.append([0.0, 0.0, 0.0])

            return np.array(pts)

        waypoints = generate_ccw_circle_points()

        # waypoints = np.array([ # x, y, yaw

        #     # [0.0, 0.0, 0.0], # SQUARE
        #     # [5.75, 0.0, 0.0],
        #     # [5.75, 5.75, 0.0],
        #     # [0.0, 5.75, 0.0],
        #     # [0.0, 0.0, 0.0],


        #     [-1.0, 3.0, 0.0], # HW SQUARE
        #     [-1.0, 5.0, 0.0],
        #     [1.0, 5.0, 0.0],
        #     [1.0, 3.0, 0.0],
        #     [-1.0, 3.0, 0.0],

        #     # [origin,    origin,    0.0], # SQUARE
        #     # [max_point, origin,    0.0],
        #     # [max_point, max_point, 0.0],
        #     # [origin,    max_point, 0.0],
        #     # [origin,    origin,    0.0],
            
        #     # # Back and Forth
        #     # [0.0, 0.0, 0.0],
        #     # [3.0, 0.0, 0.0],
        #     # [1.0, 1.0, 0.0],
        #     # [1.0, 0.0, 0.0],
        #     # [0.0, 0.0, 0.0],

        #     # [0.0, 0.0, 0.0], # TRIANGLE
        #     # [9.0, 4.5, 0.0],
        #     # [0.0, 9.0, 0.0],
        #     # [0.0, 0.0, 0.0],
        # ])
        
        self.trajectory = WaypointTraj(waypoints, v_lin=v_lin, w_yaw=w_yaw)
    
    def callback_state_error(self, msg: StateError):
        self.err_xy = math.hypot(msg.err_x, msg.err_y)
        self.err_yaw = msg.err_yaw
    
    def reference_update(self):
        if not self.begun:
            self.begun = True
            self.get_logger().info("Beginning trajectory tracking.")
            self.last_reference_time = self.get_clock().now()

            # Publish waypoints as Path for visualization
            path_msg = Path()
            path_msg.header.frame_id = "odom"
            path_msg.header.stamp = self.get_clock().now().to_msg()

            for pt in self.trajectory.points:
                x, y, yaw = float(pt[0]), float(pt[1]), float(pt[2])

                ps = PoseStamped()
                ps.header.frame_id = "odom"
                ps.header.stamp = path_msg.header.stamp  # keep a consistent stamp
                ps.pose.position.x = x
                ps.pose.position.y = y
                ps.pose.position.z = 0.0

                ps.pose.orientation.x = 0.0
                ps.pose.orientation.y = 0.0
                ps.pose.orientation.z = math.sin(yaw * 0.5)
                ps.pose.orientation.w = math.cos(yaw * 0.5)

                path_msg.poses.append(ps)
            self.waypoints_path_pub_.publish(path_msg)

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

    # Used if we want to change parameter during runtime
    def parameters_callback(self, params: list[Parameter]): 
        for p in params:
            if p.name == "v_lin":
                self.trajectory.v_lin = p.value
                self.get_logger().info(f"{p.name} changed to {p.value}")
            elif p.name == "w_yaw":
                self.trajectory.w_yaw = p.value
                self.get_logger().info(f"{p.name} changed to {p.value}")
            elif p.name == "reference_timer_hz":
                self.reference_timer_hz = p.value
                self.reference_timer_.cancel()
                self.reference_timer_ = self.create_timer(
                    1 / self.reference_timer_hz, self.reference_update)
                self.get_logger().info(f"{p.name} changed to {p.value}")
            
class WaypointTraj(object):
    def __init__(self, points, v_lin=0.6, w_yaw=0.3):
        """
        Inputs: points, (N, 3) array of N waypoint coordinates in 2D with yaw
        """
        points = np.array(points, dtype=float)

        # Keep points properly shaped
        if points.ndim == 1:
            if points.size % 3 != 0:
                raise ValueError("points.size % 3 != 0")
            points = points.reshape(-1, 3)
        elif points.ndim == 3 and points.shape[1] != 3:
            if points.shape[0] == 3:
                points = points.T
            else:
                raise ValueError("points must be (N, 3) or (3, N)")

        self.points = points
        self.v_lin = float(v_lin)
        self.w_yaw = float(w_yaw)
        self.N = len(points)

        def wrap_angle(a):
            return np.arctan2(np.sin(a), np.cos(a))

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