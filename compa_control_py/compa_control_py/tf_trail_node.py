#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose

from tf2_msgs.msg import TFMessage # to access TFs
import tf_transformations
from geometry_msgs.msg import Quaternion

class TfTrailNode(Node):
    def __init__(self):
        super().__init__("tf_trail_node")

        # -------- Parameters --------
        self.declare_parameter("odom_topic", "/compa/odom")
        self.declare_parameter("marker_topic", "/tf_trail")
        self.declare_parameter("frame_id", "odom")            # RViz fixed frame for markers
        self.declare_parameter("lifetime_sec", 70.0)           # how long each arrow stays
        self.declare_parameter("min_dist", 0.1)              # meters between drops
        self.declare_parameter("drop_every_n_msgs", 1)        # optional: sub-sample odom

        self.declare_parameter("arrow_shaft_len", 0.25)       # Marker.scale.x
        self.declare_parameter("arrow_shaft_diam", 0.03)      # Marker.scale.y
        self.declare_parameter("arrow_head_diam", 0.06)       # Marker.scale.z
        self.declare_parameter("color_rgba", [0.47, 0.0, 0.784, 1.0])  # r,g,b,a

        self.odom_topic       = self.get_parameter("odom_topic").get_parameter_value().string_value
        self.marker_topic     = self.get_parameter("marker_topic").get_parameter_value().string_value
        self.frame_id         = self.get_parameter("frame_id").get_parameter_value().string_value
        self.lifetime_sec     = self.get_parameter("lifetime_sec").get_parameter_value().double_value
        self.min_dist         = self.get_parameter("min_dist").get_parameter_value().double_value
        self.drop_every_n     = self.get_parameter("drop_every_n_msgs").get_parameter_value().integer_value
        self.arrow_len        = self.get_parameter("arrow_shaft_len").get_parameter_value().double_value
        self.arrow_diam       = self.get_parameter("arrow_shaft_diam").get_parameter_value().double_value
        self.head_diam        = self.get_parameter("arrow_head_diam").get_parameter_value().double_value
        r,g,b,a               = self.get_parameter("color_rgba").get_parameter_value().double_array_value
        self.color = (float(r), float(g), float(b), float(a))

        # -------- ROS I/O --------
        self.odom_sub = self.create_subscription(Odometry, self.odom_topic, self.odom_cb, 20)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        # -------- State --------
        self.last_drop_xy = None
        self.seq_id = 0
        self.msg_counter = 0

        self.get_logger().info(
            f"TF trail node: listening to {self.odom_topic}, publishing arrows to {self.marker_topic}"
        )

        self.tf_sub_ = self.create_subscription(TFMessage, "/tf", self.callback_tf, 10)
        self.tf_static_sub = self.create_subscription(TFMessage, "/tf_static", self.callback_tf, 10)

        self._t_base_roll  = None
        self._t_roll_pitch = None
        self._t_pitch_yaw  = None

        self.q_w_y = [0.0, 0.0, 0.0, 1.0]
        self.yaw_link_base_orientation_ = Quaternion()
        self.yaw_link_base_orientation_.w = 1.0
        self.pose_base_ = Pose()
        self.pose_base_.orientation.w = 1.0

    @staticmethod
    def _dist2(p, q):
        dx = p[0] - q[0]
        dy = p[1] - q[1]
        return dx*dx + dy*dy

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

        q_b_p = tf_transformations.quaternion_multiply(q_b_r, q_r_p) # base to pitch
        q_b_y = tf_transformations.quaternion_multiply(q_b_p, q_p_y) # base to yaw
        q_b_y = self._quat_normalized(q_b_y)

        q_w_b = [self.pose_base_.orientation.x,
                self.pose_base_.orientation.y,
                self.pose_base_.orientation.z,
                self.pose_base_.orientation.w]
        
        self.q_w_y = tf_transformations.quaternion_multiply(q_w_b, q_b_y)
        self.yaw_link_base_orientation_.x = self.q_w_y[0]
        self.yaw_link_base_orientation_.y = self.q_w_y[1]
        self.yaw_link_base_orientation_.z = self.q_w_y[2]
        self.yaw_link_base_orientation_.w = self.q_w_y[3]

    def odom_cb(self, msg: Odometry):
        self.msg_counter += 1
        if self.drop_every_n > 1 and (self.msg_counter % self.drop_every_n) != 0:
            return

        # Pose in odom
        px = msg.pose.pose.position.x
        py = msg.pose.pose.position.y
        self.pose_base_ = msg.pose.pose

        if self.last_drop_xy is not None:
            if self._dist2((px, py), self.last_drop_xy) < (self.min_dist * self.min_dist):
                return

        # Create an ARROW marker at this exact pose (position+orientation from odom)
        arrow = Marker()
        arrow.header.frame_id = self.frame_id
        arrow.header.stamp = msg.header.stamp  # keep the original stamp to align with TF time if needed
        arrow.ns = "tf_trail"
        arrow.id = self.seq_id
        arrow.type = Marker.ARROW
        arrow.action = Marker.ADD

        # Use the odom pose directly (assumes odom->base_link orientation reflects heading)
        arrow.pose.position = msg.pose.pose.position
        arrow.pose.orientation = self.yaw_link_base_orientation_

        # Geometry
        arrow.scale.x = float(self.arrow_len)      # shaft length
        arrow.scale.y = float(self.arrow_diam)     # shaft diameter
        arrow.scale.z = float(self.head_diam)      # head diameter

        # Color
        arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = self.color

        # Lifetime (auto-expire)
        arrow.lifetime = Duration(sec=int(self.lifetime_sec),
                                  nanosec=int((self.lifetime_sec % 1.0) * 1e9))

        # Optional: make the marker view-independent? (usually false for TF arrows)
        arrow.frame_locked = False  # keep it at fixed pose in frame

        # Publish as a single-element MarkerArray (cheaper than managing DELETEs)
        marr = MarkerArray()
        marr.markers.append(arrow)
        self.marker_pub.publish(marr)

        # Update state
        self.last_drop_xy = (px, py)
        self.seq_id += 1


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
