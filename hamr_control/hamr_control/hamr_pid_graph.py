''' Live Graphing of ODOM and PID
A bit slower than only odom (multithreading was too slow as well, 
both reenrant and multithreaded executor)
'''

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from tf2_msgs.msg import TFMessage
from hamr_interfaces.msg import ReferenceTraj, LiveGains

import math
import matplotlib.pyplot as plt
import time

### - - UTILITIES - - ###
def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def quat_to_angle(q):
    return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z))

# ---------- Node 1: Odom/TF/Reference ----------
class OdomGraphNode(Node):
    def __init__(self):
        super().__init__("hamr_live_graph_node")
        # self.odom_sub_ = self.create_subscription(
            # Odometry, "/hamr/odom", self.callback_odom, 10
        # )
        # self.tf_sub_ = self.create_subscription(
            # TFMessage, "/tf", self.callback_tf, 10
        # )
        # self.reference_sub_ = self.create_subscription(
            # ReferenceTraj, "/reference_trajectory", self.callback_reference, 10
        # )
        self.gains_sub_ = self.create_subscription(
            LiveGains, "/live_gains", self.callback_gains, 10
        )
        
        self.get_logger().info("HamrLiveGraphNode started.")
        
        # current values
        # self.curr_x = 0.0
        # self.curr_y = 0.0
        # self.curr_yaw_b_w = 0.0
        # self.curr_yaw_t_b = 0.0

        # self.reference_x = 0.0
        # self.reference_y = 0.0
        # self.reference_yaw = 0.0

        self.live_gains = dict(
            p_x=0.0, i_x=0.0, d_x=0.0,
            p_y=0.0, i_y=0.0, d_y=0.0,
            p_yaw=0.0, i_yaw=0.0, d_yaw=0.0
        )

    def callback_odom(self, msg: Odometry):
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

    def callback_gains(self, msg: LiveGains):
        self.live_gains["p_x"] = float(msg.p_x)
        self.live_gains["i_x"] = float(msg.i_x)
        self.live_gains["d_x"] = float(msg.d_x)
        self.live_gains["p_y"] = float(msg.p_y)
        self.live_gains["i_y"] = float(msg.i_y)
        self.live_gains["d_y"] = float(msg.d_y)
        self.live_gains["p_yaw"] = float(msg.p_yaw)
        self.live_gains["i_yaw"] = float(msg.i_yaw)
        self.live_gains["d_yaw"] = float(msg.d_yaw)

def main(args=None):
    rclpy.init(args=args)
    node = OdomGraphNode()

    ### Fig 1: Odometry
    plt.ion()
    # fig1, ax1 = plt.subplots()
    # ax1.set_xlabel('Time (s)')
    # ax1.set_ylabel('Value')
    # ax1.set_title('Odometry: x, y, yaw')
    # line_x,   = ax1.plot([], [], label='x', color='blue', linewidth=2)
    # line_y,   = ax1.plot([], [], label='y', color='green', linewidth=2)
    # line_yaw, = ax1.plot([], [], label='yaw', color='red', linewidth=2)
    # line_x_ref,   = ax1.plot([], [], label='x_ref', color='blue', linewidth=1, linestyle='dashed')
    # line_y_ref,   = ax1.plot([], [], label='y_ref', color='green', linewidth=1, linestyle='dashed')
    # line_yaw_ref, = ax1.plot([], [], label='yaw_ref', color='red', linewidth=1, linestyle='dashed')
    # ax1.legend()

    ### Fig 2: Live PID Terms
    fig2, (axx, axy, axyaw) = plt.subplots(3, 1, sharex=True)
    fig2.canvas.manager.set_window_title('Live PID Terms')
    axyaw.set_xlabel('Time (s)')

    for ax, title in zip((axx, axy, axyaw), ('PID terms: X', 'PID terms: Y', 'PID terms: Yaw')):
        ax.set_ylabel('Value')
        ax.set_title(title)

    ## P/I/D lines for each DOF
    # X
    lx_px,   = axx.plot([], [], label='P_x', linewidth=2)
    lx_ix,   = axx.plot([], [], label='I_x', linewidth=2)
    lx_dx,   = axx.plot([], [], label='D_x', linewidth=2)
    lx_zero, = axx.plot([], [], linestyle='dashed', linewidth=1, label='ref=0')  # zero reference
    axx.legend(loc='upper left')

    # Y
    ly_py,   = axy.plot([], [], label='P_y', linewidth=2)
    ly_iy,   = axy.plot([], [], label='I_y', linewidth=2)
    ly_dy,   = axy.plot([], [], label='D_y', linewidth=2)
    ly_zero, = axy.plot([], [], linestyle='dashed', linewidth=1, label='ref=0')
    axy.legend(loc='upper left')

    # Yaw
    lz_pyaw,   = axyaw.plot([], [], label='P_yaw', linewidth=2)
    lz_iyaw,   = axyaw.plot([], [], label='I_yaw', linewidth=2)
    lz_dyaw,   = axyaw.plot([], [], label='D_yaw', linewidth=2)
    lz_zero,   = axyaw.plot([], [], linestyle='dashed', linewidth=1, label='ref=0')
    axyaw.legend(loc='upper left')

    # data buffers
    t_buf, x_buf, y_buf, yaw_buf = [], [], [], []
    # x_buf_ref, y_buf_ref, yaw_buf_ref = [], [], []

    # buffers for PID terms
    px_buf, ix_buf, dx_buf = [], [], []
    py_buf, iy_buf, dy_buf = [], [], []
    pyaw_buf, iyaw_buf, dyaw_buf = [], [], []

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
            # x_buf.append(node.curr_x)
            # y_buf.append(node.curr_y)
            # yaw_buf.append(wrap_angle(node.curr_yaw_b_w + node.curr_yaw_t_b))
            # x_buf_ref.append(node.reference_x)
            # y_buf_ref.append(node.reference_y)
            # yaw_buf_ref.append(wrap_angle(node.reference_yaw))

            # live PID terms
            px_buf.append(node.live_gains["p_x"]);   ix_buf.append(node.live_gains["i_x"]);   dx_buf.append(node.live_gains["d_x"])
            py_buf.append(node.live_gains["p_y"]);   iy_buf.append(node.live_gains["i_y"]);   dy_buf.append(node.live_gains["d_y"])
            pyaw_buf.append(node.live_gains["p_yaw"]); iyaw_buf.append(node.live_gains["i_yaw"]); dyaw_buf.append(node.live_gains["d_yaw"])

            # trim history if needed
            trim(t_buf, #x_buf, y_buf, yaw_buf, x_buf_ref, y_buf_ref, yaw_buf_ref,
                 px_buf, ix_buf, dx_buf, py_buf, iy_buf, dy_buf, pyaw_buf, iyaw_buf, dyaw_buf)

            # update ODOM lines
            # line_x.set_data(t_buf, x_buf)
            # line_y.set_data(t_buf, y_buf)
            # line_yaw.set_data(t_buf, yaw_buf)
            # line_x_ref.set_data(t_buf, x_buf_ref)
            # line_y_ref.set_data(t_buf, y_buf_ref)
            # line_yaw_ref.set_data(t_buf, yaw_buf_ref)
            # ax1.relim(); ax1.autoscale_view()

            # --- update GAINS lines ---
            zeros = [0.0] * len(t_buf)

            # X terms
            lx_px.set_data(t_buf, px_buf)
            lx_ix.set_data(t_buf, ix_buf)
            lx_dx.set_data(t_buf, dx_buf)
            lx_zero.set_data(t_buf, zeros)
            axx.relim(); axx.autoscale_view()

            # Y terms
            ly_py.set_data(t_buf, py_buf)
            ly_iy.set_data(t_buf, iy_buf)
            ly_dy.set_data(t_buf, dy_buf)
            ly_zero.set_data(t_buf, zeros)
            axy.relim(); axy.autoscale_view()

            # Yaw terms
            lz_pyaw.set_data(t_buf, pyaw_buf)
            lz_iyaw.set_data(t_buf, iyaw_buf)
            lz_dyaw.set_data(t_buf, dyaw_buf)
            lz_zero.set_data(t_buf, zeros)
            axyaw.relim(); axyaw.autoscale_view()

            plt.pause(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
    
    
if __name__ == "__main__":
    main()