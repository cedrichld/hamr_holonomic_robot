import math
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy # not used yet

from std_msgs.msg import Float64 # to send velocity commands
from nav_msgs.msg import Odometry # used to get the base current state (position in xyz)
from geometry_msgs.msg import PoseWithCovariance # used for reference and current pose - not using covariance rn
from tf2_msgs.msg import TFMessage # to access TFs (for turret relative angle) - could also be used for position esimation with "encoders"
from geometry_msgs.msg import Quaternion # for the turret relative 

''' Main issues '''
## FIXED: Using wrong yaw in Jacobian (and in pid_controller?)
    # Fix1: Subscribe to JointState (or TF) to read the turret joint angle
    # Fix2: Use yaw_drive = yaw_base + turret_angle in the Jacobian
## TODO: Controller timing (dt)
    # Integrating using fixed dt but call from odom callback which 
        # has its own rate (I/D will be way off)
    # Fix: compute dt from msg.header.stamp or run timer

''' Smaller issues '''
## TODO:Cap velocitie in x,y and yaw as well as joint velocities 

### - - UTILITIES - - ###
def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

def quat_to_angle(q):
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

class HamrControlNode(Node):
    def __init__(self):
        super().__init__("hamr_controller_node")

        ### - - HAMR Config params (m) - - ###
        default_hamr_config = {"r_wheel": 0.0762,
                               "a_wheel": 0.149556,
                               "b_wheel": 0.19682}
        for a, b in default_hamr_config.items():
            self.declare_parameter(a, b)
        self.hamr_config = {
            "r_wheel": self.get_parameter("r_wheel").value,
            "a_wheel": self.get_parameter("a_wheel").value,
            "b_wheel": self.get_parameter("b_wheel").value,
        }

        ### - - dt Settings - - ###
        self.declare_parameter("dt", 0.05)
        self.dt = self.get_parameter("dt").value
        
        ### - - PID Parameters for x, y and yaw - - ###
        PID_default_gains = {
            "P_x": 5.0, "I_x": 3.0, "D_x": 0.05,
            "P_y": 5.0, "I_y": 3.0, "D_y": 0.05,
            "P_yaw": 10.0, "I_yaw": 1.5, "D_yaw": 0.2,
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
            "yaw": {
                "P" : self.get_parameter("P_yaw").value,
                "I" : self.get_parameter("I_yaw").value,
                "D" : self.get_parameter("D_yaw").value,
            }
        }

        self.add_post_set_parameters_callback(self.parameters_callback)
        # self.timer_ = self.create_timer(self.dt, self.publish_cmd)

        ### - - Set Publishers and Subscribers - - ##
        self.left_wheel_vel_ = self.create_publisher(Float64, "/left_wheel/cmd_vel", 1)
        self.right_wheel_vel_ = self.create_publisher(Float64, "/right_wheel/cmd_vel", 1)
        self.turret_vel_ = self.create_publisher(Float64, "/turret/cmd_vel", 1)
        
        self.odom_sub_ = self.create_subscription(Odometry, "/hamr/odom", self.callback_odom, 1)
        self.tf_sub_ = self.create_subscription(TFMessage, "/tf", self.callback_tf, 1)

        self.reference_sub_ = self.create_subscription(PoseWithCovariance, "/reference_trajectory", 
                                    self.callback_reference, 1)

        ## - - State Variables - - ##        
        self.pose_base_: PoseWithCovariance = None # interested in x, y, yaw
        self.reference_: PoseWithCovariance = None # interested in x, y, yaw
        self.turret_to_base_orientation_: Quaternion = None # interested in relative yaw of turret

        self.err_x_prev = 0.0
        self.err_y_prev = 0.0
        self.err_yaw_prev = 0.0

        ## - - Filtered derivatives - - ##
        self.d_err_x_filt = 0.0
        self.d_err_y_filt = 0.0
        self.d_err_yaw_filt = 0.0
        self.d_alpha = 0.15 # 0 < alpha < 1 (lower stronger smoothing)

        ## - - Integral Accumulators - - ##
        self.I_x = PIAccumulator(limit=5.0)
        self.I_y = PIAccumulator(limit=5.0)
        self.I_yaw = PIAccumulator(limit=2.0)

        ## - - thresholds - - ##
        self.threshold_x_y = 0.01
        self.threshold_yaw = 0.01

        self.get_logger().info("HAMR Controller has been started with P_x: " + str(self.gains["x"]["P"]) + 
                               ", I_x: " + str(self.gains["x"]["I"]) + ", D_x: " + str(self.gains["x"]["D"])
                                + "; P_y: " + str(self.gains["y"]["P"]) + 
                               ", I_y: " + str(self.gains["y"]["I"]) + ", D_x: " + str(self.gains["y"]["D"])
                                + "; P_yaw: " + str(self.gains["yaw"]["P"]) + ", I_yaw: " + 
                                str(self.gains["yaw"]["I"]) + ", D_yaw: " + str(self.gains["yaw"]["D"]))

    def pid_step(self):
        ''' Compute velocities based on PID Controller Logic '''
        def compute_errors():
            ''' Find the distance error to target '''
            err_x = self.reference_.pose.position.x - self.pose_base_.pose.position.x
            err_y = self.reference_.pose.position.y - self.pose_base_.pose.position.y

            yaw_des = quat_to_angle(self.reference_.pose.orientation) # desired yaw for the turret wrt to world frame (used for error)
            yaw_curr_b_w = quat_to_angle(self.pose_base_.pose.orientation) # base orientation wrt to world frame (used for error)
            yaw_curr_t_b = quat_to_angle(self.turret_to_base_orientation_) # turret orientation wrt to base (used for error AND used in Jac)
            yaw_curr_t_w = yaw_curr_b_w + yaw_curr_t_b # turret orientation wrt to world frame (used for error)

            err_yaw = wrap_angle(yaw_des - yaw_curr_t_w)

            return err_x, err_y, err_yaw, yaw_curr_t_b # yaw_curr_t_b passed to jacobian later
        
        if self.pose_base_ == None:
            self.get_logger().warn("Waiting on odom to publish cmds")
            return
        if self.reference_ == None:
            # self.get_logger().info("Waiting on target to publish cmds")
            return
        if self.turret_to_base_orientation_ == None:
            # self.get_logger().info("Waiting on target to publish cmds")
            return

        err_x, err_y, err_yaw, yaw_curr_t_b = compute_errors()
        
        ## x, y loop
        if math.hypot(err_x, err_y) < self.threshold_x_y:
            ## Check if at target
            desired_x_dot, desired_y_dot = 0, 0
            self.err_x_prev = 0
            self.err_y_prev = 0
            self.I_x.reset()
            self.I_y.reset()
        else:
            # - for x - #
            P_x = self.gains["x"]["P"] * err_x
            I_x_term = self.gains["x"]["I"] * self.I_x.update(err_x, self.dt)

            d_raw_x = (err_x - self.err_x_prev) / self.dt
            self.d_err_x_filt = (self.d_alpha * d_raw_x +
                                (1.0 - self.d_alpha) * self.d_err_x_filt)
            D_x = self.gains["x"]["D"] * self.d_err_x_filt

            desired_x_dot = P_x + I_x_term + D_x
            self.err_x_prev = err_x

            # - for y - #
            P_y = self.gains["y"]["P"] * err_y
            I_y_term = self.gains["y"]["I"] * self.I_y.update(err_y, self.dt)

            d_raw_y = (err_y - self.err_y_prev) / self.dt
            self.d_err_y_filt = (self.d_alpha * d_raw_y +
                                (1.0 - self.d_alpha) * self.d_err_y_filt)
            D_y = self.gains["y"]["D"] * self.d_err_y_filt

            desired_y_dot = P_y + I_y_term + D_y
            self.err_y_prev = err_y
            
        ## yaw loop
        if abs(err_yaw) < self.threshold_yaw:
            ## Check if at target
            desired_yaw_dot = 0 
            self.err_yaw_prev = 0
            self.I_yaw.reset()
        else:
            P_yaw = self.gains["yaw"]["P"] * err_yaw
            I_yaw_term = self.gains["yaw"]["I"] * self.I_yaw.update(err_yaw, self.dt)

            d_raw_yaw = (err_yaw - self.err_yaw_prev) / self.dt
            self.d_err_yaw_filt = (self.d_alpha * d_raw_yaw +
                                (1.0 - self.d_alpha) * self.d_err_yaw_filt)
            D_yaw = self.gains["yaw"]["D"] * self.d_err_yaw_filt

            desired_yaw_dot = P_yaw + I_yaw_term + D_yaw
            self.err_yaw_prev = err_yaw
        
        self.publish_joint_cmd(np.array([desired_x_dot, desired_y_dot, 
                                        desired_yaw_dot]), wrap_angle(yaw_curr_t_b)) # desired vel

    def callback_odom(self, msg: Odometry):
        ''' Subscription callback to the pose of turtle1 '''
        self.pose_base_ = msg.pose
        self.pid_step()

    def callback_tf(self, msg: TFMessage):
        ''' Look through all TFs and find turret_link to get it's Quaternion '''
        for t in msg.transforms:
            if t.child_frame_id == "turret_link":
                self.turret_to_base_orientation_ = t.transform.rotation # Quaternion
                break

    def callback_reference(self, msg: PoseWithCovariance):
        self.reference_ = msg
        self.I_x.reset()
        self.I_y.reset()
        self.I_yaw.reset()
        self.get_logger().info("Going to target: " + str((msg.pose.position.x, msg.pose.position.y)))

    def compute_velocities(self, desired_velocity, yaw):
        ''' Derived Jacobian based on dynamics - returns angular velocities for:
                1. right_wheel
                2. left_wheel
                3. turret 
        '''
        r_w, b, a = self.hamr_config["r_wheel"], \
            self.hamr_config["b_wheel"], self.hamr_config["a_wheel"]
        c, s = np.cos(yaw), np.sin(yaw)
        
        J = np.array([
            [r_w/2 * (c + s*b/a), r_w/2 * (c - s*b/a), 0],
            [r_w/2 * (-s + c*b/a), r_w/2 * (-s - c*b/a), 0],
            [r_w/(2*a), -r_w/(2*a), 1]
        ])

        return np.linalg.solve(J, desired_velocity) # will return angular vels for joints

    def publish_joint_cmd(self, desired_velocity, yaw):
        right_wheel_omega, left_wheel_omega, turret_omega = Float64(), Float64(), Float64()
        omegas = self.compute_velocities(desired_velocity, yaw)
        right_wheel_omega.data, left_wheel_omega.data, turret_omega.data = omegas

        self.right_wheel_vel_.publish(right_wheel_omega)
        self.left_wheel_vel_.publish(left_wheel_omega)
        self.turret_vel_.publish(turret_omega)

    # Used if we want to change parameter during runtime
    def parameters_callback(self, params: list[Parameter]): 
        pid_name_map = {
            "P_x": ("x", "P"),
            "I_x": ("x", "I"),
            "D_x": ("x", "D"),
            "P_y": ("y", "P"),
            "I_y": ("y", "I"),
            "D_y": ("y", "D"),
            "P_yaw":("yaw", "P"),
            "I_yaw":("yaw", "I"),
            "D_yaw":("yaw", "D"),
        }
        config_name_map = ("r_wheel", "a_wheel", "b_wheel")
        for p in params:
            if p.name in pid_name_map:
                group, term = pid_name_map[p.name]
                self.gains[group][term] = p.value
                self.get_logger().info(f"{p.name} changed to {p.value}")
            elif p.name in config_name_map:
                self.hamr_config[p.name] = p.value
                self.get_logger().info(f"{p.name} changed to {p.value}")
            # elif p.name == "dt":
            #     new_period = float(p.value)
            #     if new_period <= 0.0:
            #         self.get_logger().warn("spawn_period must be > 0")
            #         continue
            #     self.dt = new_period
            #     self.timer_.cancel()
            #     self.timer_ = self.create_timer(self.dt, self.publish_cmd)
            #     self.get_logger().info(f"{p.name} changed to {p.value}")

def main(args=None):
    rclpy.init(args=args)
    node = HamrControlNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    
    
if __name__ == "__main__":
    main()