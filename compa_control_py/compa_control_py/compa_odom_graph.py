#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage # to access TFs (for turret relative angle) - could also be used for position esimation with "encoders"
import tf_transformations
from geometry_msgs.msg import Quaternion

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

def _yaw_from_xyzw(q_xyzw):
    x,y,z,w = q_xyzw
    # same formula you use elsewhere
    return math.atan2(2.0 * (w*z + x*y), 1.0 - 2.0 * (y*y + z*z))

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
        self.curr_yaw_b_w = Quaternion()
        self.curr_yaw_t_b = 0.0

        self._t_base_roll  = None
        self._t_roll_pitch = None
        self._t_pitch_yaw  = None

        self.reference_x = 0.0
        self.reference_y = 0.0
        self.reference_yaw = 0.0

        self.q_b_y = [0.0, 0.0, 0.0, 1.0]

    def odom_callback(self, msg: Odometry):
        self.curr_x = msg.pose.pose.position.x
        self.curr_y = msg.pose.pose.position.y
        self.curr_yaw_b_w = msg.pose.pose.orientation
    
    def callback_reference(self, msg: ReferenceTraj):
        self.reference_x = msg.x
        self.reference_y = msg.y
        self.reference_yaw = msg.yaw

    def _quat_normalized(self, q_xyzw):
        x, y, z, w = q_xyzw
        n = math.sqrt(x*x + y*y + z*z + w*w)
        if n < 1e-12:
            # fallback to I if something degenerate arrives
            return [0.0, 0.0, 0.0, 1.0]
        inv = 1.0 / n
        return [x*inv, y*inv, z*inv, w*inv]
    
    def _q_from_tf(self, t):
        return [t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w]    
    
    def callback_tf(self, msg: TFMessage):
        ''' Look through all TFs and find turret_link to get it's Quaternion '''
        for t in msg.transforms:
            if t.header.frame_id == "base_link" and t.child_frame_id == "roll_link":
                self._t_base_roll = t
            elif t.header.frame_id == "roll_link" and t.child_frame_id == "pitch_link":
                self._t_roll_pitch = t
            elif t.header.frame_id == "pitch_link" and t.child_frame_id == "yaw_plate_link":
                self._t_pitch_yaw = t

        if not (self._t_base_roll and self._t_roll_pitch and self._t_pitch_yaw):
            return

        q_b_r = self._q_from_tf(self._t_base_roll)
        q_r_p = self._q_from_tf(self._t_roll_pitch)
        q_p_y = self._q_from_tf(self._t_pitch_yaw)

        q_b_p = tf_transformations.quaternion_multiply(q_b_r, q_r_p)
        q_b_y = tf_transformations.quaternion_multiply(q_b_p, q_p_y)

        # normalize
        x,y,z,w = q_b_y
        n = math.sqrt(x*x + y*y + z*z + w*w)
        if n < 1e-12:
            q_b_y = [0.0,0.0,0.0,1.0]
        else:
            q_b_y = [x/n, y/n, z/n, w/n]

        self.q_b_y = q_b_y[:]
        self.curr_yaw_t_b = _yaw_from_xyzw(q_b_y)

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
            # yaw_buf.append(wrap_angle(node.curr_yaw_b_w + node.curr_yaw_t_b))

            # build world->base and base->turret quats
            q_w_b = [0,0,0,0]
            q_w_b[0] = math.sin(0)  # placeholder to avoid style lints
            q_w_b = [node.curr_yaw_b_w.x,
                    node.curr_yaw_b_w.y,
                    node.curr_yaw_b_w.z,
                    node.curr_yaw_b_w.w]  # use your last odom quat here
            # you already have base->turret yaw (node.curr_yaw_t_b), so synthesize a z-yaw quat:
            cz, sz = math.cos(node.curr_yaw_t_b*0.5), math.sin(node.curr_yaw_t_b*0.5)
            q_b_y = [0.0, 0.0, sz, cz]
            q_w_y = tf_transformations.quaternion_multiply(q_w_b, q_b_y)
            yaw_w = math.atan2(2*(q_w_y[3]*q_w_y[2] + q_w_y[0]*q_w_y[1]),
                            1 - 2*(q_w_y[1]*q_w_y[1] + q_w_y[2]*q_w_y[2]))
            yaw_buf.append(wrap_angle(yaw_w))

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