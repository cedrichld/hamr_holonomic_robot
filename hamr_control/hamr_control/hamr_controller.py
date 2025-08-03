import math
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovariance
# from tf2_msgs.msg import TFMessage

### - - UTILITIES - - ###
def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi

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

        ### - - Timer Settings - - ###
        self.declare_parameter("dt", 0.05)
        self.dt = self.get_parameter("dt").value
        
        ### - - PID Parameters for distance and angle - - ###
        PID_default_gains = {
            "P_x": 3.5, "I_x": 0.05, "D_x": 0.05,
            "P_y": 3.5, "I_y": 0.05, "D_y": 0.05,
            "P_yaw": 20.0, "I_yaw": 0.001, "D_yaw": 0.01,
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
        # self.sub_reference_ = self.create_subscription(Pose, "target_turtle", 
        #                             self.callback_target, 1)

        ## - - State Variables - - ##        
        self.odom_: PoseWithCovariance = None # interested in x, y, yaw
        self.target_: PoseWithCovariance = None # interested in x, y, yaw

        self.err_x_prev = 0.0
        self.err_y_prev = 0.0
        self.err_yaw_prev = 0.0

        ## - - Filtered derivatives - - ##
        self.d_err_x_filt = 0.0
        self.d_err_yaw_filt = 0.0
        self.d_alpha = 0.15 # 0 < alpha < 1 (lower stronger smoothing)

        ## - - Integral Accumulators - - ##
        self.I_d = PIAccumulator(limit=2.0)
        self.I_angle = PIAccumulator(limit=1.0)

        ## - - Velocities to publish and thresholds - - ##
        self.v_ = 0.0
        self.w_ = 0.0
        self.threshold_x_y = 0.01
        self.threshold_yaw = 0.01

        self.get_logger().info("HAMR Controller has been started with P_x: " + str(self.gains["x"]["P"]) + 
                               ", I_x: " + str(self.gains["x"]["I"]) + ", D_x: " + str(self.gains["x"]["D"])
                                + "; P_y: " + str(self.gains["y"]["P"]) + 
                               ", I_x: " + str(self.gains["y"]["I"]) + ", D_x: " + str(self.gains["y"]["D"])
                                + "; P_yaw: " + str(self.gains["yaw"]["P"]) + ", I_yaw: " + 
                                str(self.gains["yaw"]["I"]) + ", D_yaw: " + str(self.gains["yaw"]["D"]))

    def compute_errors(self):
        ''' Find the distance error to target '''
        dx = self.target_.x - self.pose_.x
        dy = self.target_.y - self.pose_.y
        err_d = math.hypot(dx, dy)

        desired_theta = math.atan2(dy, dx)
        err_angle = wrap_angle(desired_theta - self.pose_.theta)

        return err_d, err_angle

    def pid_step(self):
        ''' Compute velocity an omega based on PID Controller Logic '''
        if self.pose_ == None or self.target_ == None:
            return

        err_d, err_angle = self.compute_errors()
        
        if (err_d < self.threshold_x_y):
            ## Check if at target
            self.v_ = 0
            self.err_x_prev = 0
            self.err_y_prev = 0
            self.I_d.reset()
            if (err_angle < self.threshold_yaw):
                self.w_ = 0
                self.err_yaw_prev = 0
                self.I_angle.reset()
        else:
            ## Distance loop
            P_d = self.gains["distance"]["P"] * err_d
            I_d_term = self.gains["distance"]["I"] * self.I_d.update(err_d, self.dt)

            d_raw_d = (err_d - self.err_x_prev) / self.dt
            self.d_err_x_filt = (self.d_alpha * d_raw_d +
                                (1.0 - self.d_alpha) * self.d_err_x_filt)
            D_d = self.gains["distance"]["D"] * self.d_err_x_filt

            self.v_ = P_d + I_d_term + D_d
            self.err_x_prev = err_d

            ## Angle loop
            P_angle = self.gains["angle"]["P"] * err_angle
            I_angle_term = self.gains["angle"]["I"] * self.I_angle.update(err_angle, self.dt)

            d_raw_angle = (err_angle - self.err_yaw_prev) / self.dt
            self.d_err_yaw_filt = (self.d_alpha * d_raw_angle +
                                (1.0 - self.d_alpha) * self.d_err_yaw_filt)
            D_angle = self.gains["angle"]["D"] * self.d_err_yaw_filt

            self.w_ = P_angle + I_angle_term + D_angle
            self.err_yaw_prev = err_angle

    def callback_pose(self, msg: Pose):
        ''' Subscription callback to the pose of turtle1 '''
        self.pose_ = msg

    def callback_target(self, msg: Pose):
        self.target_ = msg
        self.I_d.reset()
        self.I_angle.reset()
        self.get_logger().info("Going to target: " + str((msg.x, msg.y)))

    def publish_velocities(self):
        left, right, turret = Float64(), Float64(), Float64()
        left.data, right.data, turret.data = pass

        self.left_wheel_vel_.publish(left)
        self.right_wheel_vel_.publish(right)
        self.turret_vel_.publish(turret)

    # Used if we want to change parameter during runtime
    def parameters_callback(self, params: list[Parameter]): 
        name_map = {
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
        for p in params:
            if p.name in name_map:
                group, term = name_map[p.name]
                self.gains[group][term] = p.value
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