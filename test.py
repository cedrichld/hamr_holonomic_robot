#!/usr/bin/env python3
# Live overlay plot: Vicon (/HAMR_Base/odom) vs Odom (/odom)
# - GUI runs in MAIN THREAD (Qt/Tk). ROS spins in a background thread.
# - Auto SE(2) alignment (2D Procrustes). Optional origin align.
# - With ssh -X/-Y, ensure an X server is running on your client.

import argparse, os, sys, threading, math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from nav_msgs.msg import Odometry

# ----------------- Backend selection (popup window) -----------------
import matplotlib
# matplotlib.use("TkAgg")

# if os.environ.get("DISPLAY", "") == "":
#     # No X display -> headless
matplotlib.use("Agg")
# else:
#     # Prefer Qt if available; otherwise Tk
#     try:
#         matplotlib.use("Qt5Agg")
#     except Exception:
#         try:
#             matplotlib.use("TkAgg")
#         except Exception:
#             matplotlib.use("Agg")

import matplotlib.pyplot as plt

def yaw_from_quat(x, y, z, w):
    return math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))

def procrustes_se2(X, Y):
    """
    Best-fit SE(2) transform T: Y ≈ R*X + t
    X, Y: (2,N) arrays. Returns R(2x2), t(2,), success(bool).
    """
    if X.shape[1] < 3 or Y.shape[1] < 3:
        return np.eye(2), np.zeros(2), False
    x_c = X.mean(axis=1, keepdims=True)
    y_c = Y.mean(axis=1, keepdims=True)
    X0 = X - x_c
    Y0 = Y - y_c
    S = X0 @ Y0.T
    U, _, Vt = np.linalg.svd(S)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, 1] *= -1
        R = U @ Vt
    t = (y_c - R @ x_c).reshape(2)
    return R, t, True

class PathPlotter(Node):
    def __init__(self, vicon_topic, odom_topic, max_pts,
                 origin_align, auto_align, align_every_n):
        super().__init__('plot_paths_align_node')

        # Buffers (thread-safe via self._lock)
        self.vx, self.vy = deque(maxlen=max_pts), deque(maxlen=max_pts)
        self.ox, self.oy = deque(maxlen=max_pts), deque(maxlen=max_pts)
        self._lock = threading.Lock()

        # Origin align flags
        self.origin_align = origin_align
        self.auto_align = auto_align
        self.align_every_n = max(50, int(align_every_n))
        self.sample_count_since_align = 0

        # Anchors for origin alignment
        self.vx0 = None; self.vy0 = None
        self.ox0 = None; self.oy0 = None

        # SE(2) transform (odom -> vicon)
        self.R = np.eye(2)
        self.t = np.zeros(2)
        self.have_T = False

        # ROS subscribers
        self.create_subscription(Odometry, vicon_topic, self.cb_vicon, 50)
        self.create_subscription(Odometry, odom_topic,  self.cb_odom,  50)

        # Matplotlib plot (must be created in main thread; we only define attrs here)
        self.fig = None
        self.ax = None
        self.l_v = None
        self.l_o = None

    # ---------------------- ROS Callbacks (background thread) ----------------------

    def _maybe_set_origin(self, is_vicon, x, y):
        if is_vicon:
            if self.vx0 is None:
                self.vx0, self.vy0 = x, y
        else:
            if self.ox0 is None:
                self.ox0, self.oy0 = x, y

    def cb_vicon(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        with self._lock:
            self._maybe_set_origin(True, x, y)
            if self.origin_align and self.vx0 is not None:
                x -= self.vx0; y -= self.vy0
            self.vx.append(x); self.vy.append(y)
            self.sample_count_since_align += 1

    def cb_odom(self, msg: Odometry):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        with self._lock:
            self._maybe_set_origin(False, x, y)
            if self.origin_align and self.ox0 is not None:
                x -= self.ox0; y -= self.oy0
            self.ox.append(x); self.oy.append(y)
            self.sample_count_since_align += 1

    # ---------------------- GUI-side helpers (main thread) ------------------------

    def _compute_auto_alignment_locked(self, vx, vy, ox, oy):
        # Use overlapping tail
        n = min(len(vx), len(ox))
        if n < 30:
            return
        V = np.vstack([np.array(vx)[-n:], np.array(vy)[-n:]])
        O = np.vstack([np.array(ox)[-n:], np.array(oy)[-n:]])
        if n > 2000:
            idx = np.linspace(0, n-1, 2000).astype(int)
            V = V[:, idx]; O = O[:, idx]
        R, t, ok = procrustes_se2(O, V)  # Odom -> Vicon
        if ok:
            self.R, self.t = R, t
            self.have_T = True

    def gui_setup(self, vicon_topic, odom_topic):
        # Called from main thread
        self.fig, self.ax = plt.subplots()
        self.l_v, = self.ax.plot([], [], label=f'Vicon ({vicon_topic})', linewidth=2)
        self.l_o, = self.ax.plot([], [], label=f'Odom  ({odom_topic})', linewidth=2)
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.grid(True, linestyle='--', linewidth=0.5)
        self.ax.set_xlabel('x[m]')
        self.ax.set_ylabel('y[m]')
        self.ax.set_title('HAMR Path Overlay (Vicon vs Odom, aligned)')
        self.ax.legend(loc='best')

    def gui_update(self):
        # This is called by Matplotlib's timer in the main thread.
        with self._lock:
            vx = list(self.vx); vy = list(self.vy)
            ox = list(self.ox); oy = list(self.oy)
            need_align = self.auto_align and (self.sample_count_since_align >= self.align_every_n)
            if need_align:
                self._compute_auto_alignment_locked(vx, vy, ox, oy)
                self.sample_count_since_align = 0

            if self.have_T and len(ox) > 0:
                O = np.vstack([np.array(ox), np.array(oy)])  # (2,N)
                OA = (self.R @ O) + self.t.reshape(2,1)
                ox_a = OA[0, :].tolist()
                oy_a = OA[1, :].tolist()
            else:
                ox_a = ox; oy_a = oy

        # Update artists
        self.l_v.set_data(vx, vy)
        self.l_o.set_data(ox_a, oy_a)

        # Autoscale
        allx = vx + ox_a
        ally = vy + oy_a
        if allx and ally:
            xmin, xmax = min(allx), max(allx)
            ymin, ymax = min(ally), max(ally)
            dx = max(0.2, 0.1*(xmax-xmin) if xmax>xmin else 0.5)
            dy = max(0.2, 0.1*(ymax-ymin) if ymax>ymin else 0.5)
            self.ax.set_xlim(xmin-dx, xmax+dx)
            self.ax.set_ylim(ymin-dy, ymax+dy)

        self.fig.canvas.draw_idle()
        # Optional: always save a snapshot too
        self.fig.savefig("hamr_paths.png", dpi=150)

# ------------------------------ Main --------------------------------

def main():
    ap = argparse.ArgumentParser(description='Popup plot: Vicon vs Odom with SE(2) alignment.')
    ap.add_argument('--vicon_topic', default='/HAMR_Base/odom',
                    help='Vicon Odometry topic (nav_msgs/Odometry)')
    ap.add_argument('--odom_topic',  default='/odom',
                    help='Robot Odometry topic (nav_msgs/Odometry)')
    ap.add_argument('--max_pts', type=int, default=40000,
                    help='Max points to keep per path')
    ap.add_argument('--origin_align', action='store_true',
                    help='Translate each path so its first point is at (0,0)')
    ap.add_argument('--no_auto_align', action='store_true',
                    help='Disable SE(2) best-fit alignment')
    ap.add_argument('--align_every_n', type=int, default=300,
                    help='Recompute SE(2) every N new samples')
    ap.add_argument('--rate_hz', type=float, default=10.0,  # GUI refresh (used for timer)
                    help='Plot refresh rate (GUI timer)')
    args = ap.parse_args()

    rclpy.init()
    node = PathPlotter(
        vicon_topic=args.vicon_topic,
        odom_topic=args.odom_topic,
        max_pts=args.max_pts,
        origin_align=args.origin_align,
        auto_align=(not args.no_auto_align),
        align_every_n=args.align_every_n
    )

    # Start ROS in a background thread
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        # GUI setup & timer (MAIN THREAD)
        node.gui_setup(args.vicon_topic, args.odom_topic)

        # Use Matplotlib's timer so updates happen on the GUI event loop
        interval_ms = max(10, int(1000.0 / max(1e-6, args.rate_hz)))
        timer = node.fig.canvas.new_timer(interval=interval_ms)
        timer.add_callback(node.gui_update)
        timer.start()

        # Blocking show in MAIN THREAD
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
