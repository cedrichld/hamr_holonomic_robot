import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

from nav_msgs.msg import Odometry # used to get the base current state (position in xyz)
from geometry_msgs.msg import PoseWithCovariance # used for reference and current pose - not using covariance rn
from geometry_msgs.msg import Quaternion # for the turret relative 
from geometry_msgs.msg import Twist # for manual mode
from tf2_msgs.msg import TFMessage
import tf_transformations

import rclpy
import tf2_ros
from builtin_interfaces.msg import Time
from rclpy.time import Time as RclpyTime

def _quat_normalized(q_xyzw):
    x, y, z, w = q_xyzw
    n = math.sqrt(x*x + y*y + z*z + w*w)
    if n < 1e-12:
        # fallback to I if something degenerate arrives
        return [0.0, 0.0, 0.0, 1.0]
    inv = 1.0 / n
    return [x*inv, y*inv, z*inv, w*inv]

def _q_from_tf(t):
    return [t.transform.rotation.x,
            t.transform.rotation.y,
            t.transform.rotation.z,
            t.transform.rotation.w]    



def get_tf_from_bag(bag_dir,
                    target_frame="map",
                    source_frame="yaw_plate_link"):

    # Accept both "foo" and "foo.bag"
    if bag_dir.endswith(".bag") and os.path.isdir(bag_dir[:-4]) and not os.path.exists(bag_dir):
        bag_dir = bag_dir[:-4]

    if not os.path.exists(bag_dir):
        raise FileNotFoundError(
            f"Bag path not found: '{bag_dir}'. "
            f"Tip: ROS2 bags are usually directories (e.g. 'compa_square_attitude')."
        )

    converter_options = ConverterOptions('', '')
    reader = SequentialReader()

    # Try storage backends in order
    last_err = None
    for storage_id in ("mcap", "sqlite3"):
        try:
            storage_options = StorageOptions(uri=bag_dir, storage_id=storage_id)
            reader.open(storage_options, converter_options)
            break
        except Exception as e:
            last_err = e
            reader = SequentialReader()  # reset
    else:
        raise RuntimeError(
            f"Could not open bag '{bag_dir}' with mcap or sqlite3. Last error: {last_err}"
        )

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    # Initialize tf2 buffer
    rclpy.init()
    buffer = tf2_ros.Buffer()

    times = []
    x_vals, y_vals = [], []
    rpy_vals = []

    while reader.has_next():
        topic, data, t_ns = reader.read_next()

        if topic not in ['/tf', '/tf_static']:
            continue

        msg_class = get_message(topic_types[topic])
        msg = deserialize_message(data, msg_class)

        # Feed TF into buffer
        for transform in msg.transforms:
            if topic == "/tf_static":
                buffer.set_transform_static(transform, "bag")
            else:
                buffer.set_transform(transform, "bag")

        # Try lookup at this timestamp
        try:
            tf = buffer.lookup_transform(
                target_frame,
                source_frame,
                RclpyTime()
            )
        except Exception:
            continue

        x = tf.transform.translation.x
        y = tf.transform.translation.y

        q = tf.transform.rotation
        roll, pitch, yaw = tf_transformations.euler_from_quaternion(
            [q.x, q.y, q.z, q.w]
        )

        times.append(t_ns * 1e-9)
        x_vals.append(x)
        y_vals.append(y)
        rpy_vals.append((roll, pitch, yaw))

    rclpy.shutdown()
    return times, x_vals, y_vals, rpy_vals

def get_tf_series_from_bag(
    bag_dir,
    world_frame="map",          # or "odom"
    base_frame="base_footprint",
    gimbal_frame="yaw_plate_link"
):
    storage_options = StorageOptions(uri=bag_dir, storage_id='mcap')
    converter_options = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    rclpy.init()
    buffer = tf2_ros.Buffer()

    times = []
    base_xyz = []    # [(x,y,z), ...]
    gimbal_xyz = []  # [(x,y,z), ...]
    gimbal_rpy = []  # [(r,p,y), ...]

    world_frame  = world_frame.lstrip("/")
    base_frame   = base_frame.lstrip("/")
    gimbal_frame = gimbal_frame.lstrip("/")

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in ["/tf", "/tf_static"]:
            continue

        msg_class = get_message(topic_types[topic])
        msg = deserialize_message(data, msg_class)

        for tr in msg.transforms:
            if topic == "/tf_static":
                buffer.set_transform_static(tr, "bag")
            else:
                buffer.set_transform(tr, "bag")

        # Sample at "latest" time to avoid stamp alignment issues
        try:
            tf_w_b = buffer.lookup_transform(world_frame, base_frame, RclpyTime())
            tf_w_g = buffer.lookup_transform(world_frame, gimbal_frame, RclpyTime())
        except Exception:
            continue

        # time (use bag time just as x-axis label)
        times.append(t_ns * 1e-9)

        # base xyz
        tb = tf_w_b.transform.translation
        base_xyz.append((tb.x, tb.y, tb.z))

        # gimbal xyz
        tg = tf_w_g.transform.translation
        gimbal_xyz.append((tg.x, tg.y, tg.z))

        # gimbal rpy
        q = tf_w_g.transform.rotation
        r, p, y = tf_transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        gimbal_rpy.append((r, p, y))

    rclpy.shutdown()
    return np.asarray(times), np.asarray(base_xyz), np.asarray(gimbal_xyz), np.asarray(gimbal_rpy)


def wrap_angle(a):
    # Wrap to [-pi, pi]
    return (a + np.pi) % (2*np.pi) - np.pi

def angle_unwrap(seq):
    # unwrap via incremental shortest-diff
    out = np.zeros_like(seq)
    out[0] = seq[0]
    for i in range(1, len(seq)):
        out[i] = out[i-1] + wrap_angle(seq[i] - seq[i-1])
    return out

def angle_diff(a, b):
    # Shortest signed difference a - b
    return wrap_angle(a - b)

def generate_ccw_circle_points(radius=5.0, steps_between=10):
    cx = 0.0
    cy = 0.0 + radius

    # Angles for waypoints (rad)
    # waypoints = [-np.pi/2, -np.pi, -3*np.pi/2, -2*np.pi, -5*np.pi/2] # CW
    waypoints = [-5*np.pi/2, -2*np.pi, -3*np.pi/2, -np.pi, -np.pi/2] # CCW
    pts = []

    # First point explicitly at (0,0,0)
    pts.append([cx, cy - radius])

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
            pts.append([float(x), float(y)])

    # Close the loop back to start
    pts.append([0.0, 0.0])

    return np.array(pts)


def main():
    bag_dir = 'canonical/circle_pid'
    times, x_vals, y_vals, rpy_vals = get_tf_from_bag(bag_dir)

    if not x_vals or not y_vals:
        print(f"No TF data found in {bag_dir}")
        return
    
    print(f"Loaded {bag_dir}: {len(x_vals)} x, {len(y_vals)} y, {len(times)} times")

    # --------- Choose reference path ----------
    # waypoints = np.array([[0,0],[5.75,0],[5.75,5.75],[0,5.75],[0,0]])  # SQUARE
    # waypoints = np.array([[0,0],[8.0, 4.0],[0.0, 8.0],[0,0]])  # Triangle
    waypoints = generate_ccw_circle_points()  # CIRCLE

    # Densify the path for nearest-neighbor distance & heading lookup
    ref_path = []
    for i in range(len(waypoints)-1):
        edge = np.linspace(waypoints[i], waypoints[i+1], 50, endpoint=False)
        ref_path.append(edge)
    ref_path = np.vstack(ref_path)

    # ---------- Position RMSE ----------
    x_run = np.asarray(x_vals, dtype=float)
    y_run = np.asarray(y_vals, dtype=float)
    pts   = np.column_stack((x_run, y_run))
    tree  = cKDTree(ref_path)
    dists, idx = tree.query(pts)  # also get indices to map headings
    pos_rmse = np.sqrt(np.mean(dists**2))

    # ---------- Extract RPY from bag ----------
    rpy = np.asarray(rpy_vals, dtype=float)
    if rpy.ndim != 2 or rpy.shape[1] != 3:
        raise ValueError("rpy_vals must be shape (N,3) as [roll, pitch, yaw].")

    x_run = np.asarray(x_vals, dtype=float)
    y_run = np.asarray(y_vals, dtype=float)
    N = min(len(x_run), len(y_run), len(rpy))
    roll, pitch, yaw = rpy[:N,0], rpy[:N,1], rpy[:N,2]

    # ---------- Errors vs zero (target = 0 for all axes) ----------
    roll_err  = wrap_angle(roll)
    pitch_err = wrap_angle(pitch)
    yaw_err   = wrap_angle(yaw)

    # ---------- RMSE and STD of errors ----------
    roll_rmse  = float(np.sqrt(np.mean(roll_err**2)))
    pitch_rmse = float(np.sqrt(np.mean(pitch_err**2)))
    yaw_rmse   = float(np.sqrt(np.mean(yaw_err**2)))

    roll_std   = float(np.std(roll_err))
    pitch_std  = float(np.std(pitch_err))
    yaw_std    = float(np.std(yaw_err))

    # (Optional) bias/mean error — useful to spot constant offsets
    roll_bias  = float(np.mean(roll_err))
    pitch_bias = float(np.mean(pitch_err))
    yaw_bias   = float(np.mean(yaw_err))

    roll_max  = float(np.max(np.abs(roll_err)))
    pitch_max = float(np.max(np.abs(pitch_err)))
    yaw_max   = float(np.max(np.abs(yaw_err)))

    ######################
    times, base_xyz, gimbal_xyz, gimbal_rpy = get_tf_series_from_bag(
        bag_dir=bag_dir,
        world_frame="map",
        base_frame="base_footprint",
        gimbal_frame="yaw_plate_link"
    )

    x_run = base_xyz[:,0]
    y_run = base_xyz[:,1]
    z_base = base_xyz[:,2]
    z_gimbal = gimbal_xyz[:,2]

    # ---------- Roughness comparison: base vs gimbal
    # VERTICAL ROUGHNESS
    def moving_average(x, w):
        w = max(1, int(w))
        k = np.ones(w) / w
        return np.convolve(x, k, mode="same")

    def roughness_rms(signal, fs, window_s=0.5):
        # detrend by subtracting moving average (low-pass)
        w = int(window_s * fs)
        trend = moving_average(signal, w)
        residual = signal - trend
        return float(np.sqrt(np.mean(residual**2))), residual

    # estimate sampling frequency from times
    dt = np.median(np.diff(times))
    fs = 1.0 / dt if dt > 1e-9 else 1.0

    base_z_rms, base_z_res = roughness_rms(z_base, fs, window_s=0.5)
    gimb_z_rms, gimb_z_res = roughness_rms(z_gimbal, fs, window_s=0.5)

    # ANGULAR ROUGHNESS (ROLL/PITCH) AT GIMBAL
    roll = angle_unwrap(gimbal_rpy[:,0])
    pitch = angle_unwrap(gimbal_rpy[:,1])

    roll_rate = np.diff(roll) / np.diff(times)
    pitch_rate = np.diff(pitch) / np.diff(times)

    roll_rate_rms = float(np.sqrt(np.mean(roll_rate**2)))
    pitch_rate_rms = float(np.sqrt(np.mean(pitch_rate**2)))

    # ---------- Elevation / slope metrics (base) ----------
    dx = np.diff(x_run)
    dy = np.diff(y_run)
    dz = np.diff(z_base)

    ds = np.sqrt(dx*dx + dy*dy)  # horizontal distance increments

    eps = 1e-9
    valid = ds > eps
    grade = dz[valid] / ds[valid]   # dz per meter traveled (unitless)

    # Elevation range (m)
    z_range = float(np.max(z_base) - np.min(z_base))

    # Grade metrics (unitless). Multiply by 100 for percent.
    mean_abs_grade = float(np.mean(np.abs(grade))) if grade.size else float("nan")
    rms_grade = float(np.sqrt(np.mean(grade**2))) if grade.size else float("nan")
    max_abs_grade = float(np.max(np.abs(grade))) if grade.size else float("nan")

    # ---------- Convert to deg ---------
    rad2deg = 180.0 / np.pi

    roll_bias_deg  = roll_bias  * rad2deg
    pitch_bias_deg = pitch_bias * rad2deg
    yaw_bias_deg   = yaw_bias   * rad2deg

    roll_std_deg   = roll_std   * rad2deg
    pitch_std_deg  = pitch_std  * rad2deg
    yaw_std_deg    = yaw_std    * rad2deg

    roll_max_deg   = roll_max   * rad2deg
    pitch_max_deg  = pitch_max  * rad2deg
    yaw_max_deg    = yaw_max    * rad2deg

    # ---------- Print summary ----------
    # print(f"\nPosition RMSE: {pos_rmse:.4f} m")
    # rad2deg = 180.0/np.pi
    # print("\nOrientation errors vs 0 (rad / deg):")
    # print(f"  roll:  RMSE={roll_rmse:.4f} ({roll_rmse*rad2deg:.2f}°), "
    #       f"STD={roll_std:.4f} ({roll_std*rad2deg:.2f}°), "
    #       f"Bias={roll_bias:+.4f} ({roll_bias*rad2deg:+.2f}°)")
    # print(f"  pitch: RMSE={pitch_rmse:.4f} ({pitch_rmse*rad2deg:.2f}°), "
    #       f"STD={pitch_std:.4f} ({pitch_std*rad2deg:.2f}°), "
    #       f"Bias={pitch_bias:+.4f} ({pitch_bias*rad2deg:+.2f}°)")
    # print(f"  yaw:   RMSE={yaw_rmse:.4f} ({yaw_rmse*rad2deg:.2f}°), "
    #       f"STD={yaw_std:.4f} ({yaw_std*rad2deg:.2f}°), "
    #       f"Bias={yaw_bias:+.4f} ({yaw_bias*rad2deg:+.2f}°)")
    
    
    # print(f"Vertical roughness RMS (detrended z): base={base_z_rms:.4f} m, gimbal={gimb_z_rms:.4f} m")
    # print(f"Roughness ratio (gimbal/base) = {gimb_z_rms/max(base_z_rms,1e-9):.2f}")
    # print(f"Gimbal angular rate RMS: roll={roll_rate_rms:.4f} rad/s, pitch={pitch_rate_rms:.4f} rad/s")

    print("\n===== Physical Experiment Metrics =====")
    print(f"Path RMSE (m): {pos_rmse:.3f}")

    print("\nRoll (deg):")
    print(f"  Bias: {roll_bias_deg:.2f}")
    print(f"  Std:  {roll_std_deg:.2f}")
    print(f"  Max:  {roll_max_deg:.2f}")

    print("\nPitch (deg):")
    print(f"  Bias: {pitch_bias_deg:.2f}")
    print(f"  Std:  {pitch_std_deg:.2f}")
    print(f"  Max:  {pitch_max_deg:.2f}")

    print("\nYaw (deg):")
    print(f"  Bias: {yaw_bias_deg:.2f}")
    print(f"  Std:  {yaw_std_deg:.2f}")
    print(f"  Max:  {yaw_max_deg:.2f}")


    print("\nElevation metrics (base_footprint):")
    print(f"  z_range: {z_range:.3f} m")
    print(f"  mean|dz/ds|: {mean_abs_grade*100:.2f} %")
    print(f"  rms(dz/ds):  {rms_grade*100:.2f} %")
    print(f"  max|dz/ds|:  {max_abs_grade*100:.2f} %")
    
    #------------------------------------------
    
    # plt.figure()
    # plt.plot(times, z_base, label="base_footprint z")
    # plt.xlabel("time (s)")
    # plt.ylabel("z (m)")
    # plt.title("Altitude (base_footprint in world frame)")
    # plt.grid(True)
    # plt.legend()
    # plt.show()
    
    # PLOT VERTICAL ROUGHNESS
    # plt.figure()
    # plt.plot(times, base_z_res, label="base detrended z")
    # plt.plot(times, gimb_z_res, label="gimbal detrended z")
    # plt.xlabel("time (s)")
    # plt.ylabel("z residual (m)")
    # plt.title("High-frequency vertical motion (roughness proxy)")
    # plt.grid(True)
    # plt.legend()
    # plt.show()

    # PLOT HEIGHT
    plt.figure()
    plt.plot(times, z_base - z_base[0], label="base_footprint Δz")
    plt.xlabel("time (s)")
    plt.ylabel("Δz (m)")
    plt.title("Altitude change (base_footprint)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ---------- Plots (optional) ----------
    plt.figure(figsize=(8,8))
    plt.plot(x_run[:N], y_run[:N], label='Path', linewidth=3)
    plt.plot(ref_path[:,0], ref_path[:,1], 'r--', label='Ideal Path', linewidth=2)
    plt.xlabel('x (m)'); plt.ylabel('y (m)')
    plt.title('hamr_sim_square: Path vs Ideal Path')
    plt.legend(); plt.grid(True); plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

if __name__ == "__main__":
    main()
