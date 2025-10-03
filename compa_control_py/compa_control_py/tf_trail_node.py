#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
import tf2_ros

class TfTrailNode(Node):
    def __init__(self):
        super().__init__("tf_trail_node")

        # -------- Parameters --------
        self.declare_parameter("odom_topic", "/compa/odom")
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("lifetime_sec", 10.0)
        self.declare_parameter("min_dist", 0.1)
        self.declare_parameter("drop_every_n_msgs", 1)

        self.odom_topic   = self.get_parameter("odom_topic").value
        self.frame_id     = self.get_parameter("frame_id").value
        self.lifetime_sec = float(self.get_parameter("lifetime_sec").value)
        self.min_dist     = float(self.get_parameter("min_dist").value)
        self.drop_every_n = int(self.get_parameter("drop_every_n_msgs").value)

        # odom and tf broadcaster
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 20)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # tf2 listener (automatic TF composition)
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=30.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)

        # State
        self.last_drop_xy = None
        self.msg_counter = 0
        self.seq_id = 0

        # Stored TF drops: {"name": str, "pos": (x,y,z), "quat": (x,y,z,w), "t0": rclpy.time.Time}
        self.drops = []
        self.broadcast_timer = self.create_timer(0.05, self._broadcast_active_drops)  # 20 Hz

        self.get_logger().info(
            f"TF trail (tf2): listening {self.odom_topic}, dropping TFs of yaw_plate_link in '{self.frame_id}'"
        )

    @staticmethod
    def _dist2(p, q):
        dx = p[0] - q[0]; dy = p[1] - q[1]
        return dx*dx + dy*dy

    def _lookup_dummy_in_world(self, stamp_msg):
        """
        Returns (pos, quat) for yaw_plate_link expressed in self.frame_id at stamp_msg.
        Falls back to latest if exact time not available.
        """
        try:
            t = self.tf_buffer.lookup_transform(
                self.frame_id, "yaw_plate_link", rclpy.time.Time.from_msg(stamp_msg))
        except Exception:
            # fallback to latest
            t = self.tf_buffer.lookup_transform(self.frame_id, "yaw_plate_link", rclpy.time.Time())

        pos = (t.transform.translation.x,
               t.transform.translation.y,
               t.transform.translation.z)
        quat = (t.transform.rotation.x,
                t.transform.rotation.y,
                t.transform.rotation.z,
                t.transform.rotation.w)
        return pos, quat

    def odom_cb(self, msg: Odometry):
        self.msg_counter += 1
        if self.drop_every_n > 1 and (self.msg_counter % self.drop_every_n) != 0:
            return

        # Gate by XY distance using base pose from odom
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        if self.last_drop_xy is not None:
            if self._dist2((px, py), self.last_drop_xy) < (self.min_dist * self.min_dist):
                return

        # Get yaw_plate_link pose in 'frame_id' via tf2
        try:
            (dx, dy, dz), (qx, qy, qz, qw) = self._lookup_dummy_in_world(msg.header.stamp)
        except Exception as e:
            self.get_logger().warn(f"lookup odom->yaw_plate_link failed: {e}")
            return

        # Record drop (pose frozen at lookup time)
        drop_name = f"gimbal_drop_{self.seq_id:05d}"
        self.seq_id += 1
        self.last_drop_xy = (px, py)     # spacing gate (based on base XY); change to (dx,dy) if preferred

        self.drops.append({
            "name": drop_name,
            "pos": (dx, dy, dz),
            "quat": (qx, qy, qz, qw),
            "t0": self.get_clock().now()
        })

    def _broadcast_active_drops(self):
        now = self.get_clock().now()
        keep = []
        for d in self.drops:
            age = (now - d["t0"]).nanoseconds * 1e-9
            if age <= self.lifetime_sec:
                t = TransformStamped()
                t.header.stamp = now.to_msg()
                t.header.frame_id = self.frame_id
                t.child_frame_id = d["name"]
                t.transform.translation.x = d["pos"][0]
                t.transform.translation.y = d["pos"][1]
                t.transform.translation.z = d["pos"][2]
                t.transform.rotation.x = d["quat"][0]
                t.transform.rotation.y = d["quat"][1]
                t.transform.rotation.z = d["quat"][2]
                t.transform.rotation.w = d["quat"][3]
                self.tf_broadcaster.sendTransform(t)
                keep.append(d)
        self.drops = keep

def main(args=None):
    rclpy.init(args=args)
    node = TfTrailNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
