#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
import tf_transformations
from geometry_msgs.msg import Quaternion

from hamr_interfaces.msg import ReferenceTraj

import math
import matplotlib.pyplot as plt
import time

### - - UTILITIES - - ###
def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def quat_to_rpy(q_xyzw):
    """Return roll, pitch, yaw (XYZ convention) from quaternion [x,y,z,w]."""
    x, y, z, w = q_xyzw

    # roll (x-axis)
    sinr_cosp = 2.0 * (w*x + y*z)
    cosr_cosp = 1.0 - 2.0 * (x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis)
    sinp = 2.0 * (w*y - z*x)
    # clamp
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    # yaw (z-axis)
    siny_cosp = 2.0 * (w*z + x*y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw

class OdomGraphNode(Node):
    def __init__(self):
        super().__init__("compa_odom_graph_node")
        self.odom_sub_ = self.create_subscription(
            Odometry, "/compa/odom", self.odom_callback, 10)
        self.tf_sub_ = self.create_subscription(
            TFMessage, "/tf", self.callback_tf, 10)
        self.tf_static_sub = self.create_subscription(
            TFMessage, "/tf_static", self.callback_tf, 10)
        self.reference_sub_ = self.create_subscription(
            ReferenceTraj, "/reference_trajectory", self.callback_reference, 1)

        self.get_logger().info("OdomGraphNode started.")

        # current values
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_turret_world = Quaternion()

        # world->gimbal chain
        self._t_world_base = None
        self._t_base_roll  = None
        self._t_roll_pitch = None
        self._t_pitch_yaw  = None

        # reference
        self.reference_x = 0.0
        self.reference_y = 0.0
        self.reference_yaw = 0.0

        # current RPY (world)
        self.curr_roll = 0.0
        self.curr_pitch = 0.0
        self.curr_yaw = 0.0

    def odom_callback(self, msg: Odometry):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self._t_world_base = msg.pose.pose.orientation

    def callback_reference(self, msg: ReferenceTraj):
        self.reference_x = msg.x
        self.reference_y = msg.y
        self.reference_yaw = msg.yaw

    def _q_from_tf(self, t):
        """Return [x,y,z,w] from either TransformStamped or Quaternion."""
        # TF TransformStamped
        if hasattr(t, "transform") and hasattr(t.transform, "rotation"):
            r = t.transform.rotation
            return [r.x, r.y, r.z, r.w]
        # geometry_msgs/Quaternion
        elif hasattr(t, "x") and hasattr(t, "y") and hasattr(t, "z") and hasattr(t, "w"):
            return [t.x, t.y, t.z, t.w]
        else:
            raise TypeError(f"Unsupported quaternion container type: {type(t)}")

    def callback_tf(self, msg: TFMessage):
        """Collect the chain odom->base->roll->pitch->yaw_plate and compute turret world quaternion."""
        for t in msg.transforms:
            if t.header.frame_id == "odom" and t.child_frame_id == "base_footprint":
                self._t_world_base = t
            elif t.header.frame_id == "base_link" and t.child_frame_id == "roll_link":
                self._t_base_roll = t
            elif t.header.frame_id == "roll_link" and t.child_frame_id == "pitch_link":
                self._t_roll_pitch = t
            elif t.header.frame_id == "pitch_link" and t.child_frame_id == "yaw_plate_link":
                self._t_pitch_yaw = t

        if not (self._t_world_base and self._t_base_roll and
                self._t_roll_pitch and self._t_pitch_yaw):
            self.get_logger().info("Waiting for full TF chain...")
            return

        q_w_b = self._q_from_tf(self._t_world_base)
        q_b_r = self._q_from_tf(self._t_base_roll)
        q_r_p = self._q_from_tf(self._t_roll_pitch)
        q_p_y = self._q_from_tf(self._t_pitch_yaw)

        # Compose: odom->yaw_plate
        q_w_r = tf_transformations.quaternion_multiply(q_w_b, q_b_r)
        q_w_p = tf_transformations.quaternion_multiply(q_w_r, q_r_p)
        q_w_y = tf_transformations.quaternion_multiply(q_w_p, q_p_y)

        # normalize
        x,y,z,w = q_w_y
        n = math.sqrt(x*x + y*y + z*z + w*w)
        if n < 1e-12:
            q_w_y = [0.0, 0.0, 0.0, 1.0]
        else:
            inv = 1.0 / n
            q_w_y = [x*inv, y*inv, z*inv, w*inv]

        # store quaternion and RPY
        self.curr_turret_world = Quaternion(x=q_w_y[0], y=q_w_y[1], z=q_w_y[2], w=q_w_y[3])
        roll, pitch, yaw = quat_to_rpy(q_w_y)
        self.curr_roll  = wrap_angle(roll)
        self.curr_pitch = wrap_angle(pitch)
        self.curr_yaw   = wrap_angle(yaw)

def main(args=None):
    rclpy.init(args=args)
    node = OdomGraphNode()

    plt.ion()

    # Figure 1: position (x, y)
    fig_pos, ax_pos = plt.subplots()
    ax_pos.set_xlabel('Time (s)')
    ax_pos.set_ylabel('Position (m)')
    ax_pos.set_title('Position: x, y')
    line_x,     = ax_pos.plot([], [], label='x', linewidth=2)
    line_y,     = ax_pos.plot([], [], label='y', linewidth=2)
    line_x_ref, = ax_pos.plot([], [], label='x_ref', linewidth=1, linestyle='dashed')
    line_y_ref, = ax_pos.plot([], [], label='y_ref', linewidth=1, linestyle='dashed')
    ax_pos.legend(loc='best')

    # Figure 2: orientation (roll, pitch, yaw)
    fig_rpy, ax_rpy = plt.subplots()
    ax_rpy.set_xlabel('Time (s)')
    ax_rpy.set_ylabel('Angle (deg)')
    ax_rpy.set_title('Orientation: roll, pitch, yaw')
    line_roll, = ax_rpy.plot([], [], label='roll', linewidth=2)
    line_pitch,= ax_rpy.plot([], [], label='pitch', linewidth=2)
    line_yaw,  = ax_rpy.plot([], [], label='yaw', linewidth=2)
    line_yaw_ref, = ax_rpy.plot([], [], label='yaw_ref', linewidth=1, linestyle='dashed')
    ax_rpy.legend(loc='best')

    # Buffers
    t_buf = []
    x_buf, y_buf = [], []
    x_buf_ref, y_buf_ref = [], []

    roll_buf, pitch_buf, yaw_buf = [], [], []
    yaw_buf_ref = []

    t0 = time.time()

    MAX_N = 300 # 30s at ~10 Hz

    def trim(*lists):
        if MAX_N is None:
            return
        for L in lists:
            if len(L) > MAX_N:
                del L[:len(L) - MAX_N]

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            t = time.time() - t0
            t_buf.append(t)

            # Position
            x_buf.append(node.curr_x)
            y_buf.append(node.curr_y)
            x_buf_ref.append(node.reference_x)
            y_buf_ref.append(node.reference_y)

            # Orientation
            roll_deg = math.degrees(node.curr_roll)
            pitch_deg = math.degrees(node.curr_pitch)
            yaw_deg = math.degrees(node.curr_yaw)
            yaw_ref_deg = math.degrees(wrap_angle(node.reference_yaw))

            roll_buf.append(roll_deg)
            pitch_buf.append(pitch_deg)
            yaw_buf.append(yaw_deg)
            yaw_buf_ref.append(wrap_angle(yaw_ref_deg))

            # Trim history by to MAX_N samples
            trim(t_buf, x_buf, y_buf, x_buf_ref, y_buf_ref,
                 roll_buf, pitch_buf, yaw_buf, yaw_buf_ref)

            # update Figure 1
            line_x.set_data(t_buf, x_buf)
            line_y.set_data(t_buf, y_buf)
            line_x_ref.set_data(t_buf, x_buf_ref)
            line_y_ref.set_data(t_buf, y_buf_ref)
            ax_pos.relim(); ax_pos.autoscale_view()

            # update Figure 2
            line_roll.set_data(t_buf, roll_buf)
            line_pitch.set_data(t_buf, pitch_buf)
            line_yaw.set_data(t_buf, yaw_buf)
            line_yaw_ref.set_data(t_buf, yaw_buf_ref)
            ax_rpy.relim(); ax_rpy.autoscale_view()

            plt.pause(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
