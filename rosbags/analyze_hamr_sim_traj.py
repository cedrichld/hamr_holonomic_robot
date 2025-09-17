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



def get_tf_from_bag(bag_dir, parent_frame="odom", child_frame="base_footprint"):
    storage_options = StorageOptions(
        uri=bag_dir,
        storage_id='mcap'
    )

    converter_options = ConverterOptions('', '')
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    x_vals = []
    y_vals = []
    roll_vals = []
    pitch_vals = []
    yaw_vals = []
    times = []
    rpy_vals = []
    topic_types = {}

    ## - - State Variables - - ##        
    pose_base_: PoseWithCovariance = None # interested in x, y, yaw
    roll_link_base_orientation_: Quaternion = None # interested in relative roll of turret
    pitch_link_base_orientation_: Quaternion = None # interested in relative pitch of turret
    yaw_link_base_orientation_: Quaternion = None # interested in relative yaw of turret
    
    # Roll Pitch Yaw TFs
    _t_base_roll  = None
    _t_roll_pitch = None
    _t_pitch_yaw  = None

    for topic in reader.get_all_topics_and_types():
        topic_types[topic.name] = topic.type

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic not in ['/tf', '/tf_static']:
            continue

        msg_type = topic_types[topic]
        msg_class = get_message(msg_type)
        msg = deserialize_message(data, msg_class)
        x, y, qx, qy, qz, qw = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for transform in msg.transforms:
            if (transform.header.frame_id == parent_frame and
                transform.child_frame_id == child_frame):
                x = transform.transform.translation.x
                y = transform.transform.translation.y
                x_vals.append(x)
                y_vals.append(y)
                times.append(t * 1e-9)
                qx = transform.transform.rotation.x
                qy = transform.transform.rotation.y
                qz = transform.transform.rotation.z
                qw = transform.transform.rotation.w
                sinr_cosp = 2 * (qw * qx + qy * qz)
                cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
                roll = math.atan2(sinr_cosp, cosr_cosp)
                sinp = 2 * (qw * qy - qz * qx)

                if abs(sinp) >= 1:
                    pitch = math.copysign(math.pi / 2, sinp)

                else:
                    pitch = math.asin(sinp)

                siny_cosp = 2 * (qw * qz + qx * qy)
                cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
                yaw = math.atan2(siny_cosp, cosy_cosp)
                # rpy_vals.append((roll, pitch, yaw))

            # base -> roll
            if transform.header.frame_id == "base_link" and transform.child_frame_id == "roll_link":
                _t_base_roll = transform
            # roll -> pitch
            elif transform.header.frame_id == "roll_link" and transform.child_frame_id == "pitch_link":
                _t_roll_pitch = transform
            # pitch -> yaw
            elif transform.header.frame_id == "pitch_link" and transform.child_frame_id == "yaw_plate_link":
                _t_pitch_yaw = transform

        if not (_t_base_roll and _t_roll_pitch and _t_pitch_yaw):
            return

        q_b_r = _q_from_tf(_t_base_roll)  # base to roll
        roll_link_base_orientation_ = Quaternion(
            x=q_b_r[0], y=q_b_r[1], z=q_b_r[2], w=q_b_r[3])

        q_r_p = _q_from_tf(_t_roll_pitch) # roll to pitch
        q_p_y = _q_from_tf(_t_pitch_yaw)  # pitch to yaw

        q_b_p = tf_transformations.quaternion_multiply(q_b_r, q_r_p) # base to pitch

        pitch_link_base_orientation_ = Quaternion(
            x=q_b_p[0], y=q_b_p[1], z=q_b_p[2], w=q_b_p[3])

        q_b_y = tf_transformations.quaternion_multiply(q_b_p, q_p_y) # base to yaw
        q_b_y = _quat_normalized(q_b_y)

        yaw_link_base_orientation_ = Quaternion(
            x=q_b_y[0], y=q_b_y[1], z=q_b_y[2], w=q_b_y[3])
        
        # world->base
        q_w_b = [qx,
                qy,
                qz,
                qw]

        # base->roll
        q_b_r = [roll_link_base_orientation_.x,
                roll_link_base_orientation_.y,
                roll_link_base_orientation_.z,
                roll_link_base_orientation_.w]
        
        # base->pitch
        q_b_p = [pitch_link_base_orientation_.x,
                pitch_link_base_orientation_.y,
                pitch_link_base_orientation_.z,
                pitch_link_base_orientation_.w]

        # base->yaw
        q_b_y = [yaw_link_base_orientation_.x,
                yaw_link_base_orientation_.y,
                yaw_link_base_orientation_.z,
                yaw_link_base_orientation_.w]

        # world->roll,pitch,yaw
        q_w_r = tf_transformations.quaternion_multiply(q_w_b, q_b_r)
        q_w_p = tf_transformations.quaternion_multiply(q_w_b, q_b_p)
        q_w_y = tf_transformations.quaternion_multiply(q_w_b, q_b_y)

        # Extract WORLD roll, pitch, yaw
        roll_w = math.atan2(
            2.0*(q_w_r[3]*q_w_r[0] + q_w_r[1]*q_w_r[2]),
            1.0 - 2.0*(q_w_r[0]*q_w_r[0] + q_w_r[1]*q_w_r[1])
        )
        pitch_w = math.asin(
            2.0*(q_w_p[3]*q_w_p[1] - q_w_p[2]*q_w_p[0])
        )
        yaw_turret_w = math.atan2(
            2.0*(q_w_y[3]*q_w_y[2] + q_w_y[0]*q_w_y[1]),
            1.0 - 2.0*(q_w_y[1]*q_w_y[1] + q_w_y[2]*q_w_y[2])
        )

        rpy_vals.append((roll_w, pitch_w, yaw_turret_w))

        

    return times, x_vals, y_vals, rpy_vals


def wrap_angle(a):
    # Wrap to [-pi, pi]
    return (a + np.pi) % (2*np.pi) - np.pi

def angle_diff(a, b):
    # Shortest signed difference a - b
    return wrap_angle(a - b)

def main():
    bag_dir = 'compa_square_attitude.bag'
    times, x_vals, y_vals, rpy_vals = get_tf_from_bag(bag_dir)

    if not x_vals or not y_vals:
        print(f"No TF data found in {bag_dir}")
        return
    
    print(f"Loaded {bag_dir}: {len(x_vals)} x, {len(y_vals)} y, {len(times)} times")

    # --------- Choose reference path ----------
    waypoints = np.array([[0,0],[5,0],[5,5],[0,5],[0,0]])  # SQUARE
    # (You can swap to your CIRCLE or MAZE definitions)

    # Densify the path for nearest-neighbor distance & heading lookup
    ref_path = []
    for i in range(len(waypoints)-1):
        edge = np.linspace(waypoints[i], waypoints[i+1], 50, endpoint=False)
        ref_path.append(edge)
    ref_path = np.vstack(ref_path)

    # ---------- Position RMSE (you already had this) ----------
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

    # ---------- Print summary ----------
    rad2deg = 180.0/np.pi
    print("\nOrientation errors vs 0 (rad / deg):")
    print(f"  roll:  RMSE={roll_rmse:.4f} ({roll_rmse*rad2deg:.2f}°), "
          f"STD={roll_std:.4f} ({roll_std*rad2deg:.2f}°), "
          f"Bias={roll_bias:+.4f} ({roll_bias*rad2deg:+.2f}°)")
    print(f"  pitch: RMSE={pitch_rmse:.4f} ({pitch_rmse*rad2deg:.2f}°), "
          f"STD={pitch_std:.4f} ({pitch_std*rad2deg:.2f}°), "
          f"Bias={pitch_bias:+.4f} ({pitch_bias*rad2deg:+.2f}°)")
    print(f"  yaw:   RMSE={yaw_rmse:.4f} ({yaw_rmse*rad2deg:.2f}°), "
          f"STD={yaw_std:.4f} ({yaw_std*rad2deg:.2f}°), "
          f"Bias={yaw_bias:+.4f} ({yaw_bias*rad2deg:+.2f}°)")

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
