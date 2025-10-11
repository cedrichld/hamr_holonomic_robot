#!/usr/bin/env python31
# File used for plotting Vicon vs EKF Odom in real time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from hamr_interfaces.msg import ReferenceTraj

import math
import matplotlib.pyplot as plt
import numpy as np

### - - UTILITIES - - ###
def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def quat_to_angle(q):
    return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

class OdomGraphNode(Node):
    def __init__(self):
        super().__init__("hamr_odom_graph_node")
        self.base_vicon_sub_ = self.create_subscription(
            Odometry, "/HAMR_Base/odom", self.base_vicon_callback_, 1)
        self.base_odom_sub_ = self.create_subscription(
            Odometry, "/odom", self.base_odom_callback_, 1)
        self.turret_vicon_sub_ = self.create_subscription(
            Odometry, "/HAMR_Turret/odom", self.turret_vicon_callback_, 1)
        self.reference_sub_ = self.create_subscription(
            ReferenceTraj, "/reference_trajectory", self.callback_reference, 1)
        
        self.get_logger().info("OdomGraphNode started.")
        
        # current values
        self.curr_x = 0.0
        self.curr_y = 0.0
        self.curr_x_est = 0.0
        self.curr_y_est = 0.0
        self.curr_yaw_t_w = 0.0

        self.init_x = 0.0
        self.init_y = 0.0
        self.init_yaw = 0.0
        self.init_pose_set = True # assume known start

        self.reference_x = 0.0
        self.reference_y = 0.0
        self.reference_yaw = 0.0

        self.waypoints = np.array([ # x, y, yaw
            [-1.0, -5.0, 0.0], # HW SQUARE
            [-1.0, -3.0, 0.0],
            [1.0, -3.0, 0.0],
            [1.0, -5.0, 0.0],])

    def base_vicon_callback_(self, msg: Odometry):
        if not self.init_pose_set:
            self.init_x = msg.pose.pose.position.x - 0.06463093098238556 # mocap markers offset
            self.init_y = msg.pose.pose.position.y + 0.04782778830030647 # mocap markers offset
            self.init_pose_set = True
            self.get_logger().info(f"Initial pose set: x={self.init_x:.3f}, y={self.init_y:.3f}, yaw={self.init_yaw:.3f} rad") 

        self.curr_x = msg.pose.pose.position.x - self.init_x # mocap markers offset
        self.curr_y = msg.pose.pose.position.y - self.init_y # mocap markers offset

    def base_odom_callback_(self, msg: Odometry):
        self.curr_x_est = msg.pose.pose.position.x
        self.curr_y_est = msg.pose.pose.position.y

    def callback_reference(self, msg: ReferenceTraj):
        self.reference_x = msg.x
        self.reference_y = msg.y
        self.reference_yaw = msg.yaw

    def turret_vicon_callback_(self, msg: Odometry):
        self.curr_yaw_t_w = quat_to_angle(msg.pose.pose.orientation)

def main(args=None):
    rclpy.init(args=args)
    node = OdomGraphNode()

    # ---- Plot config ----
    ARROW_STEP = 0.20 # meters between arrows
    ARROW_LEN = 0.16 # visual arrow length (meters)
    MAX_N = 3000 # keep last N points for the path

    plt.ion()
    fig, ax = plt.subplots()
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Base pose: XY with heading arrows')

    # Path line
    (line_vicon_path,) = ax.plot([], [], linewidth=2, label='path')
    (line_odom_path,) = ax.plot([], [], linewidth=2, label='path')
    (ref_path,)  = ax.plot([], [], linewidth=1, label='reference', color='green', linestyle='--')

    # Current-pose arrow
    curr_quiv = ax.quiver([0.0], [0.0], [0.0], [0.0],
                        angles='xy', scale_units='xy', scale=1.0)
    
    # Trail arrows every 0.1 m
    trail_quiv = ax.quiver([0.0], [0.0], [0.0], [0.0],
        angles='xy', scale_units='xy', scale=1.0,
        color='red', alpha=0.9, width=0.004, headwidth=4, headlength=6)
    
    trail_count = 0

    # DEBUG: faint points where arrows should be
    trail_pts = ax.scatter([], [], s=6, c='gray', alpha=0.5, zorder=4)

    ax.legend()
    ax.set_xlim(-3, 3)
    ax.set_ylim(2, 6)

    # --- DRAW WAYPOINT SQUARE (once) ---
    # wps = node.waypoints
    # # close the loop back to the first corner
    # xs = list(wps[:, 0]) + [wps[0, 0]]
    # ys = list(wps[:, 1]) + [wps[0, 1]]
    # (square_line,) = ax.plot(xs, ys, 'o--', linewidth=1., label='waypoints')

    # Buffers
    x_buf, y_buf, x_est_buf, y_est_buf, x_ref_buf, y_ref_buf, yaw_buf = [], [], [], [], [], [], []

    # Arrow buffers (positions and unit directions scaled by ARROW_LEN)
    arx, ary, aru, arv = [], [], [], []

    # Distance tracking for arrow placement
    last_x = None
    last_y = None
    dist_since_arrow = 0.0

    def trim_history():
        # Trim path buffers
        if MAX_N is None:
            return
        if len(x_buf) > MAX_N:
            cut = len(x_buf) - MAX_N
            del x_buf[:cut]
            del y_buf[:cut]
            del x_est_buf[:cut]
            del y_est_buf[:cut]
            del x_ref_buf[:cut]
            del y_ref_buf[:cut]
            del yaw_buf[:cut]
        # Trim arrows loosely tied to the visible window
        # (optional: keep all arrows; here we keep them bounded to ~2*MAX_N)
        MAX_A = 2 * (MAX_N // max(1, int(ARROW_STEP / 1e-6)))  # rough, safe cap
        if len(arx) > MAX_A:
            cut = len(arx) - MAX_A
            del arx[:cut]; del ary[:cut]; del aru[:cut]; del arv[:cut]
        
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

            x = node.curr_x
            y = node.curr_y
            x_est = node.curr_x_est
            y_est = node.curr_y_est
            x_ref = node.reference_x
            y_ref = node.reference_y
            yaw = wrap_angle(node.curr_yaw_t_w)

            # UPDATE current yaw arrow (position = current x,y; direction = yaw)
            curr_quiv.set_offsets(np.array([[x, y]]))
            curr_quiv.set_UVC([ARROW_LEN * math.cos(yaw)],
                            [ARROW_LEN * math.sin(yaw)])

            x_buf.append(x)
            y_buf.append(y)
            x_est_buf.append(x_est)
            y_est_buf.append(y_est)
            x_ref_buf.append(x_ref)
            y_ref_buf.append(y_ref)
            yaw_buf.append(yaw)

            # Arrow placement logic
            nonlocal_last = (last_x is None or last_y is None)
            if nonlocal_last:
                last_x, last_y = x, y
                u = ARROW_LEN * math.cos(yaw)
                v = ARROW_LEN * math.sin(yaw)
                arx.append(x); ary.append(y); aru.append(u); arv.append(v)
            else:
                dx = x - last_x
                dy = y - last_y
                ds = math.hypot(dx, dy)
                if ds > 0.0:
                    dist_since_arrow += ds
                    last_x, last_y = x, y
                    # place an arrow when we pass the step
                    if dist_since_arrow >= ARROW_STEP:
                        # Arrow points in yaw direction; make a fixed-length glyph
                        u = ARROW_LEN * math.cos(yaw)
                        v = ARROW_LEN * math.sin(yaw)
                        arx.append(x); ary.append(y); aru.append(u); arv.append(v)
                        dist_since_arrow = 0.0  # reset after placing

            # Trim
            trim_history()

            # Update artists
            line_vicon_path.set_data(x_buf, y_buf)
            line_odom_path.set_data(x_est_buf, y_est_buf)
            ref_path.set_data(x_ref_buf, y_ref_buf)
            
            # Recreate the trail quiver every frame
            if len(arx) == 0:
                # keep a harmless invisible quiver
                try:
                    trail_quiv.remove()
                except Exception:
                    pass
                trail_quiv = ax.quiver([np.nan], [np.nan], [np.nan], [np.nan],
                    angles='xy', scale_units='xy', scale=1.0,
                    color='red', alpha=0.9,
                    pivot='mid', width=0.004, headwidth=4, headlength=6,
                    zorder=5)
            else:
                try:
                    trail_quiv.remove()
                except Exception:
                    pass
                trail_quiv = ax.quiver(np.array(arx), np.array(ary),
                                    np.array(aru), np.array(arv),
                    angles='xy', scale_units='xy', scale=1.0,
                    color='red', alpha=0.9,
                    pivot='mid', width=0.004, headwidth=4, headlength=6,
                    zorder=5)

            # DEBUG: also show points where arrows should be
            if len(arx):
                trail_pts.set_offsets(np.column_stack([arx, ary]))
            else:
                trail_pts.set_offsets(np.empty((0, 2)))

            if len(arx) % 10 == 1:
                node.get_logger().info(f"trail arrows: {len(arx)}")





            # Autoscale
            ax.relim()
            ax.autoscale_view()

            plt.pause(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()