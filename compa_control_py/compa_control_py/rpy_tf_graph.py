#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import rclpy.time
import tf2_ros

import math
import time
import matplotlib
import matplotlib.pyplot as plt

def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def quat_to_rpy(q_xyzw):
    """Return roll, pitch, yaw (XYZ convention)"""
    x, y, z, w = q_xyzw
    
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    sinp = 2.0 * (w*y - z*x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

class RpyTfGraphNode(Node):
    def __init__(self):
        super().__init__("rpy_tf_graph_node")

        # Parameters
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("target_link", "yaw_plate_link")
        self.declare_parameter("sample_period", 0.05)
        self.declare_parameter("max_points", 1000)
        self.declare_parameter("plot_degrees", True)
        self.declare_parameter("warn_interval_sec", 2.0)

        self.frame_id      = self.get_parameter("frame_id").value
        self.target_link   = self.get_parameter("target_link").value
        self.sample_period = float(self.get_parameter("sample_period").value)
        self.max_points    = int(self.get_parameter("max_points").value)
        self.plot_degrees  = bool(self.get_parameter("plot_degrees").value)
        self.warn_interval = float(self.get_parameter("warn_interval_sec").value)

        # TF buffer/listener
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # Plot setup
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(9, 4.5), constrained_layout=True)
        self.ax.set_title(f"RPY of {self.target_link} in {self.frame_id}")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Angle (deg)" if self.plot_degrees else "Angle (rad)")
        (self.line_roll,)  = self.ax.plot([], [], label="roll")
        (self.line_pitch,) = self.ax.plot([], [], label="pitch")
        (self.line_yaw,)   = self.ax.plot([], [], label="yaw")
        self.ax.legend(loc="best")

        # Buffers
        self.t0 = time.time()
        self.t_hist = []
        self.roll_hist = []
        self.pitch_hist = []
        self.yaw_hist = []

        self._last_warn_time = 0.0

        self.get_logger().info(f"RPY graph: {self.frame_id} -> {self.target_link}, dt={self.sample_period}s")
        self.timer = self.create_timer(self.sample_period, self._on_timer)

    def _axes_alive(self):
        return plt.fignum_exists(self.fig.number)

    def _append(self, r, p, y):
        t = time.time() - self.t0
        self.t_hist.append(t)
        self.roll_hist.append(r)
        self.pitch_hist.append(p)
        self.yaw_hist.append(y)

        if self.max_points and len(self.t_hist) > self.max_points:
            cut = len(self.t_hist) - self.max_points
            for L in (self.t_hist, self.roll_hist, self.pitch_hist, self.yaw_hist):
                del L[:cut]

    def _rate_warn(self, msg: str):
        now = time.time()
        if now - self._last_warn_time >= self.warn_interval:
            self.get_logger().warning(msg)
            self._last_warn_time = now

    def _on_timer(self):
        # Prefer latest transform (works with sim time or wall time)
        target_time = rclpy.time.Time()

        # Ensure frames exist in buffer (prevents immediate exceptions)
        try:
            if not self.tf_buffer.can_transform(self.frame_id, self.target_link, target_time, timeout=Duration(seconds=0.2)):
                self._rate_warn(f"Waiting for TF {self.frame_id} -> {self.target_link} ...")
                return
        except Exception as e:
            self._rate_warn(f"TF can_transform error: {e}")
            return

        # Lookup- if extrapolation at a stamp occurs, retry with latest
        try:
            tfmsg = self.tf_buffer.lookup_transform(self.frame_id, self.target_link, target_time, timeout=Duration(seconds=0.2))
        except Exception as e1:
            # Retry with "latest" (Time()) if first attempt failed due to extrapolation or latency
            try:
                tfmsg = self.tf_buffer.lookup_transform(self.frame_id, self.target_link, rclpy.time.Time(), timeout=Duration(seconds=0.2))
            except Exception as e2:
                self._rate_warn(f"TF lookup failed: {e2}")
                return

        q = tfmsg.transform.rotation
        roll, pitch, yaw = quat_to_rpy((q.x, q.y, q.z, q.w))

        roll  = wrap_angle(roll)
        pitch = wrap_angle(pitch)
        yaw   = wrap_angle(yaw)

        if self.plot_degrees:
            roll  = math.degrees(roll)
            pitch = math.degrees(pitch)
            yaw   = math.degrees(yaw)

        self._append(roll, pitch, yaw)

        if not self._axes_alive():
            return

        self.line_roll.set_data(self.t_hist, self.roll_hist)
        self.line_pitch.set_data(self.t_hist, self.pitch_hist)
        self.line_yaw.set_data(self.t_hist, self.yaw_hist)
        self.ax.relim(); self.ax.autoscale_view()

        try:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)
        except Exception:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = RpyTfGraphNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
