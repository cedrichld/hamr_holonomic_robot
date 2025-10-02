import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy # not used yet

from std_msgs.msg import Float64 # to send velocity commands
from nav_msgs.msg import Odometry # used to get the base current state (position in xyz)
from geometry_msgs.msg import PoseWithCovariance # used for reference and current pose - not using covariance rn
from geometry_msgs.msg import Quaternion # for the turret relative 
from geometry_msgs.msg import Twist # for manual mode
from tf2_msgs.msg import TFMessage # to access TFs (for turret relative angle) - could also be used for position esimation with "encoders"
import tf_transformations # for quaternion operations

from hamr_interfaces.msg import LiveGains, ReferenceTraj # could create a compa interface later


### - - UTILITIES - - ###
def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def quat_to_yaw(q):
    return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

class PIAccumulator:
    def __init__(self, limit: float):
        self.sum = 0.0
        self.limit = abs(limit)

    def update(self, error: float, dt: float) -> float:
        self.sum += error * dt
        self.sum = max(-self.limit, min(self.sum, self.limit))
        return self.sum

    def reset(self):
        self.sum = 0.0

class CompaControlNode(Node):
    def __init__(self):
        super().__init__("compa_controller_node")

        ### - - COMPA Config params (m) - - ###
        default_compa_config = {"r_wheel": 0.1075,
                                "a_wheel": 0.331643,
                                "b_wheel": 0.274986, 
                                "mode": "auto"} # "auto" or "manual"
        for a, b in default_compa_config.items():
            self.declare_parameter(a, b)
        self.compa_config = {
            "r_wheel": self.get_parameter("r_wheel").value,
            "a_wheel": self.get_parameter("a_wheel").value,
            "b_wheel": self.get_parameter("b_wheel").value,
            "mode": self.get_parameter("mode").value
        }
        
        ### - - PID Parameters for x, y, roll, pitch and yaw - - ###
        PID_default_gains = {
            "P_x": 0.1, "I_x": 0.005, "D_x": 0.001,
            "P_y": 0.1, "I_y": 0.005, "D_y": 0.001,
            "P_roll": 0.1, "I_roll": 0.005, "D_roll": 0.001,
            "P_pitch": 0.1, "I_pitch": 0.005, "D_pitch": 0.001,
            "P_yaw": 0.5, "I_yaw": 0.001, "D_yaw": 0.001,
        }
        for a, b in PID_default_gains.items():
            self.declare_parameter(a, b)
        self.gains = {
            "x": {
                "P" : self.get_parameter("P_x").value,
                "I" : self.get_parameter("I_x").value,
                "D" : self.get_parameter("D_x").value,
            },
            "y": {
                "P" : self.get_parameter("P_y").value,
                "I" : self.get_parameter("I_y").value,
                "D" : self.get_parameter("D_y").value,
            },
            "roll": {
                "P" : self.get_parameter("P_roll").value,
                "I" : self.get_parameter("I_roll").value,
                "D" : self.get_parameter("D_roll").value,
            },
            "pitch": {
                "P" : self.get_parameter("P_pitch").value,
                "I" : self.get_parameter("I_pitch").value,
                "D" : self.get_parameter("D_pitch").value,
            },
            "yaw": {
                "P" : self.get_parameter("P_yaw").value,
                "I" : self.get_parameter("I_yaw").value,
                "D" : self.get_parameter("D_yaw").value,
            }
        }

        self.declare_parameter("control_rate_hz", 100.0)
        self.declare_parameter("d_alpha", 0.4)

        self.add_post_set_parameters_callback(self.parameters_callback)

        ### - - Set Publishers and Subscribers - - ##
        self.left_wheel_vel_ = self.create_publisher(Float64, "/left_wheel/cmd_vel", 1)
        self.right_wheel_vel_ = self.create_publisher(Float64, "/right_wheel/cmd_vel", 1)
        self.roll_vel_ = self.create_publisher(Float64, "/roll/cmd_vel", 1)
        self.pitch_vel_ = self.create_publisher(Float64, "/pitch/cmd_vel", 1)
        self.yaw_vel_ = self.create_publisher(Float64, "/yaw/cmd_vel", 1)
        
        self.odom_sub_ = self.create_subscription(Odometry, "/compa/odom", self.callback_odom, 1)
        self.tf_sub_ = self.create_subscription(TFMessage, "/tf", self.callback_tf, 10)
        self.tf_static_sub = self.create_subscription(TFMessage, "/tf_static", self.callback_tf, 10)

        self.reference_sub_ = self.create_subscription(ReferenceTraj, "/reference_trajectory", 
                                    self.callback_reference, 1)
        
        # For debugging
        self.gains_pub_ = self.create_publisher(LiveGains, "/live_gains", 10)
        
        # Control Rate
        self.control_rate_hz = self.get_parameter("control_rate_hz").value
        self.last_control_time = self.get_clock().now()

        if self.compa_config["mode"] == "auto":
            self.control_timer_ = self.create_timer(1.0 / self.control_rate_hz, self.control_tick)
            self.get_logger().info("Auto mode: controlling at " + str(self.control_rate_hz) + " Hz")
        elif self.compa_config["mode"] == "manual":
            self.manual_sub_ = self.create_subscription(Twist, "/cmd_vel", 
                                        self.manual_mode_callback, 1)
            self.get_logger().info("Manual mode: listening to /cmd_vel")
        self.dt = 0.0

        ### - - Variables - - ###

        ## - - State Variables - - ##        
        self.pose_base_: PoseWithCovariance = None # interested in x, y, yaw
        self.reference_: ReferenceTraj = None # interested in x, y, yaw
        self.roll_link_base_orientation_: Quaternion = None # interested in relative roll of turret
        self.pitch_link_base_orientation_: Quaternion = None # interested in relative pitch of turret
        self.yaw_link_base_orientation_: Quaternion = None # interested in relative yaw of turret

        # Roll Pitch Yaw TFs
        self._t_base_roll  = None
        self._t_roll_pitch = None
        self._t_pitch_yaw  = None

        self.err_x_prev = 0.0
        self.err_y_prev = 0.0
        self.err_pitch_prev = 0.0
        self.err_roll_prev = 0.0
        self.err_yaw_prev = 0.0

        ## - - Filtered derivatives - - ##
        self.d_err_x_filt = 0.0
        self.d_err_y_filt = 0.0
        self.d_err_roll_filt = 0.0
        self.d_err_pitch_filt = 0.0
        self.d_err_yaw_filt = 0.0
        self.d_alpha = self.get_parameter("d_alpha").value # 0 < alpha < 1 (lower stronger smoothing)

        ## - - Integral Accumulators - - ##
        self.I_x = PIAccumulator(limit=.5)
        self.I_y = PIAccumulator(limit=.5)
        self.I_roll = PIAccumulator(limit=.5)
        self.I_pitch = PIAccumulator(limit=.5)
        self.I_yaw = PIAccumulator(limit=1.0)

        ## - - Thresholds - - ##
        self.threshold_x_y = 0.005 # 0.5cm
        self.threshold_roll_pitch = 0.15 # 7 deg
        self.threshold_yaw = 0.05 # 2.86 deg

        ## - - Velocity Limits (Magnitude) - - ##
        self.xy_dot_limit = 5.0
        # 240 deg/s - equivalent to going from -30deg to 30deg in 0.5s (with constant accel)
        self.roll_pitch_dot_limit = 2.0 * math.pi * (240.0 / 360.0) 
        self.yaw_dot_limit = 2.0

        self.get_logger().info("COMPA Controller has been started with P_x: " + str(self.gains["x"]["P"]) + 
                               ", I_x: " + str(self.gains["x"]["I"]) + ", D_x: " + str(self.gains["x"]["D"])
                                + "; P_y: " + str(self.gains["y"]["P"]) + 
                               ", I_y: " + str(self.gains["y"]["I"]) + ", D_y: " + str(self.gains["y"]["D"])
                                + "; P_yaw: " + str(self.gains["yaw"]["P"]) + ", I_yaw: " + 
                                str(self.gains["yaw"]["I"]) + ", D_yaw: " + str(self.gains["yaw"]["D"]))

    def pid_step(self):
        ''' Autonomous Mode - compute velocities based on PID Controller Logic:
            - Compute errors based on pose
            - Compute desired velocities based on (a) feed-forward (b) PID corrections from pose errors
            - Feed desired velocities to jacobian (to get joint commands)
        '''
        def compute_errors(): 
            ''' Find the distance error to target '''
            err_x = self.reference_.x - self.pose_base_.pose.position.x
            err_y = self.reference_.y - self.pose_base_.pose.position.y


            roll_des = self.reference_.roll # desired roll for the turret wrt to world frame (used for error) - 0.0 for now
            pitch_des = self.reference_.pitch # desired pitch for the turret wrt to world frame (used for error) - 0.0 for now
            yaw_des = self.reference_.yaw # desired yaw for the turret wrt to world frame (used for error)
            yaw_base_w = quat_to_yaw(self.pose_base_.pose.orientation) # base orientation wrt to world frame (used for error)
            # yaw_turret_b = quat_to_yaw(self.yaw_link_base_orientation_) # turret orientation wrt to base (used for error AND used in Jac)
            
            # world->base
            q_w_b = [self.pose_base_.pose.orientation.x,
                    self.pose_base_.pose.orientation.y,
                    self.pose_base_.pose.orientation.z,
                    self.pose_base_.pose.orientation.w]

            # base->roll
            q_b_r = [self.roll_link_base_orientation_.x,
                    self.roll_link_base_orientation_.y,
                    self.roll_link_base_orientation_.z,
                    self.roll_link_base_orientation_.w]
            
            # base->pitch
            q_b_p = [self.pitch_link_base_orientation_.x,
                    self.pitch_link_base_orientation_.y,
                    self.pitch_link_base_orientation_.z,
                    self.pitch_link_base_orientation_.w]

            # base->yaw
            q_b_y = [self.yaw_link_base_orientation_.x,
                    self.yaw_link_base_orientation_.y,
                    self.yaw_link_base_orientation_.z,
                    self.yaw_link_base_orientation_.w]

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
            # roll_w = math.atan2(
            #     2.0*(q_w_y[3]*q_w_y[0] + q_w_y[1]*q_w_y[2]),
            #     1.0 - 2.0*(q_w_y[0]*q_w_y[0] + q_w_y[1]*q_w_y[1])
            # )
            # pitch_w = math.asin(
            #     2.0*(q_w_y[3]*q_w_y[1] - q_w_y[2]*q_w_y[0])
            # )
            yaw_turret_w = math.atan2(
                2.0*(q_w_y[3]*q_w_y[2] + q_w_y[0]*q_w_y[1]),
                1.0 - 2.0*(q_w_y[1]*q_w_y[1] + q_w_y[2]*q_w_y[2])
            )
            
            # yaw_turret_w = wrap_angle(yaw_base_w + yaw_turret_b) # turret orientation wrt to world frame (used for error)
            err_roll = wrap_angle(roll_des - roll_w)
            err_pitch = wrap_angle(pitch_des - pitch_w)
            err_yaw = wrap_angle(yaw_des - yaw_turret_w)

            return err_x, err_y, err_roll, err_pitch, err_yaw, yaw_base_w # yaw_base_w passed to jacobian later

        err_x, err_y, err_roll, err_pitch, err_yaw, yaw_base_w = compute_errors()

        # For debugging and publishing gains
        P_x = D_x = I_x_term = 0.0
        P_y = D_y = I_y_term = 0.0
        P_roll = D_roll = I_roll_term = 0.0
        P_pitch = D_pitch = I_pitch_term = 0.0
        P_yaw = D_yaw = I_yaw_term = 0.0

        ## X loop
        if abs(err_x) < self.threshold_x_y:
            ## Check if at target
            desired_x_dot = self.reference_.x_dot
            self.err_x_prev = err_x
            self.I_x.reset()
            # self.get_logger().warn("RESET I_x At target: " + str(self.reference_.x))
        else:
            # self.get_logger().warn("X not at target: " + str(err_x))
            P_x = self.gains["x"]["P"] * err_x
            I_x_term = self.gains["x"]["I"] * self.I_x.update(err_x, self.dt)

            d_raw_x = (err_x - self.err_x_prev) / self.dt
            self.d_err_x_filt = (self.d_alpha * d_raw_x +
                                (1.0 - self.d_alpha) * self.d_err_x_filt)
            D_x = self.gains["x"]["D"] * self.d_err_x_filt

            # Cap desired velocity
            desired_x_dot = self.reference_.x_dot + P_x + I_x_term + D_x
            self.err_x_prev = err_x
        
        ## Y loop
        if abs(err_y) < self.threshold_x_y:
            ## Check if at target
            desired_y_dot = self.reference_.y_dot
            self.err_y_prev = err_y
            self.I_y.reset()
            # self.get_logger().warn("RESET I_y At target: " + str(self.reference_.y))
        else:
            # self.get_logger().warn("Y not at target: " + str(err_y))
            P_y = self.gains["y"]["P"] * err_y
            I_y_term = self.gains["y"]["I"] * self.I_y.update(err_y, self.dt)

            d_raw_y = (err_y - self.err_y_prev) / self.dt
            self.d_err_y_filt = (self.d_alpha * d_raw_y +
                                (1.0 - self.d_alpha) * self.d_err_y_filt)
            D_y = self.gains["y"]["D"] * self.d_err_y_filt

            desired_y_dot = self.reference_.y_dot + P_y + I_y_term + D_y
            self.err_y_prev = err_y

        ## Control the XY dot NORM
        desired_xy_dot_norm = math.hypot(desired_x_dot, desired_y_dot)
        if desired_xy_dot_norm > self.xy_dot_limit:
            self.get_logger().warn("CAPPING x,y velocity from " + str(desired_xy_dot_norm) + " to " + str(self.xy_dot_limit))
            desired_x_dot = (desired_x_dot / desired_xy_dot_norm) * self.xy_dot_limit
            desired_y_dot = (desired_y_dot / desired_xy_dot_norm) * self.xy_dot_limit
        
        
        ## Roll loop
        if abs(err_roll) < self.threshold_roll_pitch:
            ## Check if at target
            desired_roll_dot = self.reference_.roll_dot
            self.err_roll_prev = err_roll
            self.I_roll.reset()
            # self.get_logger().warn("RESET I_roll At target: " + str(self.reference_.roll))
        else:
            P_roll = self.gains["roll"]["P"] * err_roll
            I_roll_term = self.gains["roll"]["I"] * self.I_roll.update(err_roll, self.dt)

            d_raw_roll = (err_roll - self.err_roll_prev) / self.dt
            self.d_err_roll_filt = (self.d_alpha * d_raw_roll +
                                (1.0 - self.d_alpha) * self.d_err_roll_filt)
            D_roll = self.gains["roll"]["D"] * self.d_err_roll_filt

            desired_roll_dot = self.reference_.roll_dot + P_roll + I_roll_term + D_roll
            self.err_roll_prev = err_roll

        ## Pitch loop
        if abs(err_pitch) < self.threshold_roll_pitch:
            ## Check if at target
            desired_pitch_dot = self.reference_.pitch_dot
            self.err_pitch_prev = err_pitch
            self.I_pitch.reset()
            # self.get_logger().warn("RESET I_pitch At target: " + str(self.reference_.pitch))
        else:
            P_pitch = self.gains["pitch"]["P"] * err_pitch
            I_pitch_term = self.gains["pitch"]["I"] * self.I_pitch.update(err_pitch, self.dt)

            d_raw_pitch = (err_pitch - self.err_pitch_prev) / self.dt
            self.d_err_pitch_filt = (self.d_alpha * d_raw_pitch +
                                (1.0 - self.d_alpha) * self.d_err_pitch_filt)
            D_pitch = self.gains["pitch"]["D"] * self.d_err_pitch_filt

            desired_pitch_dot = self.reference_.pitch_dot + P_pitch + I_pitch_term + D_pitch
            self.err_pitch_prev = err_pitch

        ## Control the XY dot NORM
        desired_rp_dot_norm = math.hypot(desired_roll_dot, desired_pitch_dot)
        if desired_rp_dot_norm > self.roll_pitch_dot_limit:
            self.get_logger().warn("CAPPING roll,pitch velocity from " + str(desired_rp_dot_norm) + " to " + str(self.roll_pitch_dot_limit))
            desired_roll_dot = (desired_roll_dot / desired_rp_dot_norm) * self.roll_pitch_dot_limit
            desired_pitch_dot = (desired_pitch_dot / desired_rp_dot_norm) * self.roll_pitch_dot_limit

        ## Yaw loop
        if abs(err_yaw) < self.threshold_yaw:
            ## Check if at target
            desired_yaw_dot = self.reference_.yaw_dot
            self.err_yaw_prev = err_yaw
            self.I_yaw.reset()
            # self.get_logger().warn("RESET I_yaw At target: " + str(self.reference_.yaw))
        else:
            P_yaw = self.gains["yaw"]["P"] * err_yaw
            I_yaw_term = self.gains["yaw"]["I"] * self.I_yaw.update(err_yaw, self.dt)

            d_raw_yaw = (err_yaw - self.err_yaw_prev) / self.dt
            self.d_err_yaw_filt = (self.d_alpha * d_raw_yaw +
                                (1.0 - self.d_alpha) * self.d_err_yaw_filt)
            D_yaw = self.gains["yaw"]["D"] * self.d_err_yaw_filt

            desired_yaw_dot = max(-self.yaw_dot_limit, min(
                self.reference_.yaw_dot + P_yaw + I_yaw_term + D_yaw, self.yaw_dot_limit))

            self.err_yaw_prev = err_yaw
        
        # self.get_logger().info(f"self.reference_.yaw_dot: {self.reference_.yaw_dot:.3f}, P_yaw: {P_yaw:.3f}, I_yaw: {I_yaw_term:.3f}, D_yaw: {D_yaw:.3f}")
        # self.get_logger().info(f"Desired x: {desired_x_dot:.3f}, y: {desired_y_dot:.3f}, yaw: {desired_yaw_dot:.3f}")
        self.publish_live_gains(P_x, D_x, I_x_term, P_y, D_y, I_y_term, P_yaw, D_yaw, I_yaw_term)
        self.publish_joint_cmd(np.array([desired_x_dot, desired_y_dot, desired_roll_dot, 
                                         desired_pitch_dot, desired_yaw_dot]), yaw_base_w)

    def manual_mode_callback(self, msg: Twist):
        ''' Manual Mode - directly compute joint commands from terminal inputs '''
        yaw_base_w = quat_to_yaw(self.pose_base_.pose.orientation)
        self.publish_joint_cmd(np.array([msg.linear.x, msg.linear.y, msg.angular.x, 
                                         msg.angular.y, msg.angular.z]), yaw_base_w)

    def publish_live_gains(self, P_x, D_x, I_x, 
                           P_y, D_y, I_y, 
                           P_yaw, D_yaw, I_yaw):
        gains = LiveGains()
        gains.p_x, gains.d_x, gains.i_x = P_x, D_x, I_x
        gains.p_y, gains.d_y, gains.i_y = P_y, D_y, I_y
        gains.p_yaw, gains.d_yaw, gains.i_yaw = P_yaw, D_yaw, I_yaw
        self.gains_pub_.publish(gains)

    def callback_odom(self, msg: Odometry):
        ''' Subscription callback to the pose of compa '''
        self.pose_base_ = msg.pose

    def control_tick(self):
        ''' Send command every (1 / control_rate_hz)[s] '''
        now = self.get_clock().now()
        dur = (now - self.last_control_time) # rclpy.duration.Duration
        self.last_control_time = now

        dt = dur.nanoseconds * 1e-9
        if not math.isfinite(dt) or dt <= 0.0:
            return
        
        self.dt = max(1e-4, min(dt, 0.1))

        if (self.pose_base_ is not None and self.reference_ is not None 
                and self.yaw_link_base_orientation_ is not None):
            self.pid_step()

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
        ''' Look through all TFs and find turret_link to get its Quaternion (wrt base link) '''
        
        for t in msg.transforms:
            # base -> roll
            if t.header.frame_id == "base_link" and t.child_frame_id == "roll_link":
                self._t_base_roll = t
            # roll -> pitch
            elif t.header.frame_id == "roll_link" and t.child_frame_id == "pitch_link":
                self._t_roll_pitch = t
            # pitch -> yaw
            elif t.header.frame_id == "pitch_link" and t.child_frame_id == "yaw_plate_link":
                self._t_pitch_yaw = t

        if not (self._t_base_roll and self._t_roll_pitch and self._t_pitch_yaw):
            return

        q_b_r = self._q_from_tf(self._t_base_roll)  # base to roll
        self.roll_link_base_orientation_ = Quaternion(
            x=q_b_r[0], y=q_b_r[1], z=q_b_r[2], w=q_b_r[3])
        
        q_r_p = self._q_from_tf(self._t_roll_pitch) # roll to pitch
        q_p_y = self._q_from_tf(self._t_pitch_yaw)  # pitch to yaw

        q_b_p = tf_transformations.quaternion_multiply(q_b_r, q_r_p) # base to pitch

        self.pitch_link_base_orientation_ = Quaternion(
            x=q_b_p[0], y=q_b_p[1], z=q_b_p[2], w=q_b_p[3])

        q_b_y = tf_transformations.quaternion_multiply(q_b_p, q_p_y) # base to yaw
        q_b_y = self._quat_normalized(q_b_y)

        self.yaw_link_base_orientation_ = Quaternion(
            x=q_b_y[0], y=q_b_y[1], z=q_b_y[2], w=q_b_y[3])

    def callback_reference(self, msg: ReferenceTraj):
        self.reference_ = msg

    def compute_velocities(self, desired_velocity, yaw):
        ''' Derived Jacobian based on dynamics - returns angular velocities for:
                1. right_wheel
                2. left_wheel
                3. turret 
        '''
        r_w, b, a = self.compa_config["r_wheel"], \
            self.compa_config["b_wheel"], self.compa_config["a_wheel"]
        c, s = np.cos(yaw), np.sin(yaw)

        J = np.array([
            [r_w/2 * (c - s*b/a), r_w/2 * (c + s*b/a), 0, 0, 0],
            [r_w/2 * (s + c*b/a), r_w/2 * (s - c*b/a), 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [r_w/(2*a), -r_w/(2*a), 0, 0, 1],
        ])

        return np.linalg.solve(J, desired_velocity) # will return angular vels for joints

    def publish_joint_cmd(self, desired_velocity, yaw):
        right_wheel_omega, left_wheel_omega, roll_omega, pitch_omega, yaw_omega = Float64(), Float64(), Float64(), Float64(), Float64()
        omegas = self.compute_velocities(desired_velocity, yaw)
        # self.get_logger().info(f"Computed omegas: {omegas}")
        right_wheel_omega.data, left_wheel_omega.data, roll_omega.data, pitch_omega.data, yaw_omega.data = omegas

        self.right_wheel_vel_.publish(right_wheel_omega)
        self.left_wheel_vel_.publish(left_wheel_omega)
        self.roll_vel_.publish(roll_omega)
        self.pitch_vel_.publish(pitch_omega)
        self.yaw_vel_.publish(yaw_omega)

    # Used if we want to change parameter during runtime
    def parameters_callback(self, params: list[Parameter]): 
        pid_name_map = {
            "P_x": ("x", "P"),
            "I_x": ("x", "I"),
            "D_x": ("x", "D"),
            "P_y": ("y", "P"),  
            "I_y": ("y", "I"),
            "D_y": ("y", "D"),
            "P_roll":("roll", "P"),
            "I_roll":("roll", "I"),
            "D_roll":("roll", "D"),
            "P_pitch":("pitch", "P"),
            "I_pitch":("pitch", "I"),
            "D_pitch":("pitch", "D"),
            "P_yaw":("yaw", "P"),
            "I_yaw":("yaw", "I"),
            "D_yaw":("yaw", "D"),
        }
        config_name_map = ("r_wheel", "a_wheel", "b_wheel", "mode", "control_rate_hz", "d_alpha")
        for p in params:
            if p.name in pid_name_map:
                group, term = pid_name_map[p.name]
                self.gains[group][term] = p.value
                self.get_logger().info(f"{p.name} changed to {p.value}")
            elif p.name in config_name_map:
                self.compa_config[p.name] = p.value
                self.get_logger().info(f"{p.name} changed to {p.value}")

def main(args=None):
    rclpy.init(args=args)
    node = CompaControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
if __name__ == "__main__":
    main()