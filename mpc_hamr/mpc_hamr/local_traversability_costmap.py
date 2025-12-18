#!/usr/bin/env python3
import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from geometry_msgs.msg import PointStamped

import tf2_ros
from tf2_ros import TransformException

from tf2_geometry_msgs import do_transform_point  # install tf2_geometry_msgs if needed


def quat_to_yaw(q):
    return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

def make_float32_multiarray(mat_2d: np.ndarray, label0="column_index", label1="row_index") -> Float32MultiArray:
    # mat_2d shape: (rows, cols)
    rows, cols = mat_2d.shape
    msg = Float32MultiArray()
    msg.layout.dim = [
        MultiArrayDimension(label=label0, size=rows, stride=rows * cols),
        MultiArrayDimension(label=label1, size=cols, stride=cols),
    ]
    msg.layout.data_offset = 0
    msg.data = mat_2d.astype(np.float32).ravel(order="F").tolist() # column-major (Eigen) - what gridmap is requiring in rviz

    return msg

def parse_layer_to_mat(layer_msg: Float32MultiArray, rows: int, cols: int) -> np.ndarray:
    # Try to reshape using layout dims if present; fallback to (rows, cols)
    data = np.array(layer_msg.data, dtype=np.float32)
    # if len(layer_msg.layout.dim) >= 2:
    #     r0 = int(layer_msg.layout.dim[0].size)
    #     c0 = int(layer_msg.layout.dim[1].size)
    #     if r0 * c0 == data.size:
    #         return data.reshape((r0, c0), order="F")
    if rows * cols != data.size:
        raise RuntimeError(f"Layer data size mismatch: expected {rows*cols}, got {data.size}")
    return data.reshape((rows, cols), order="F")

class LocalTraversabilityCost(Node):
    """
    Builds a local GridMap centered on the robot:
      - elevation_local: cropped from global elevation layer
      - cost_base: heading-free cost from slope magnitude (+ optional roughness)

    Notes:
      - Cropping is done in the GridMap message frame (grid_map_frame_id = msg.header.frame_id).
      - Output frame can be base_frame (centered at 0,0) or grid map frame (centered at robot pose).
    """

    def __init__(self):
        super().__init__("local_traversability_costmap")

        # -------------------------
        # Params
        # -------------------------
        self.input_topic = self.declare_parameter("input_topic", "/filtered_map").get_parameter_value().string_value
        self.output_topic = self.declare_parameter("output_topic", "/local_costmap").get_parameter_value().string_value

        # Which layer to treat as elevation source (must exist in incoming GridMap)
        self.elevation_layer = self.declare_parameter("elevation_layer", "elevation_inpainted").get_parameter_value().string_value

        # Local window size (meters)
        self.local_size_x = float(self.declare_parameter("local_size_x", 5.0).value)
        self.local_size_y = float(self.declare_parameter("local_size_y", 5.0).value)

        # Where to publish local map
        self.base_frame = self.declare_parameter("base_frame", "base_footprint").get_parameter_value().string_value
        self.map_frame = self.declare_parameter("map_frame", "map").value

        self.publish_in_base_frame = bool(self.declare_parameter("publish_in_base_frame", False).value)

        # Cost settings
        self.use_roughness = bool(self.declare_parameter("use_roughness", True).value)
        self.slope_max = float(self.declare_parameter("slope_max", 0.6).value) # radians-ish threshold
        self.roughness_max = float(self.declare_parameter("roughness_max", 0.05).value) # meters (tune)
        self.w_slope = float(self.declare_parameter("w_slope", 0.9).value)
        self.w_rough = float(self.declare_parameter("w_rough", 0.1).value)

        # Roughness computation radius (meters) (local std / mean abs deviation)
        self.rough_radius = float(self.declare_parameter("rough_radius", 0.3).value)

        # Debug axes handling 
        self.swap_xy = bool(self.declare_parameter("swap_xy", True).value)
        self.flip_x = bool(self.declare_parameter("flip_x", False).value)
        self.flip_y = bool(self.declare_parameter("flip_y", False).value)

        # -------------------------
        # TF
        # -------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # -------------------------
        # Dynamic Update of the map
        # -------------------------
        self.publish_rate_hz = float(self.declare_parameter("publish_rate_hz", 2.0).value)
        self.min_move_dist = float(self.declare_parameter("min_move_dist", 0.05).value)

        self.last_msg = None
        self.last_pub_xy = None

        self.timer = self.create_timer(1.0 / max(0.1, self.publish_rate_hz), self.on_timer)


        # -------------------------
        # Pub/Sub
        # -------------------------
        qos = QoSProfile(depth=1)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        self.pub = self.create_publisher(GridMap, self.output_topic, qos)
        self.sub = self.create_subscription(GridMap, self.input_topic, self.on_map, qos)

        self.get_logger().info(
            f"LocalTraversabilityCost listening on {self.input_topic}, publishing {self.output_topic} "
            f"(elevation_layer='{self.elevation_layer}')"
        )


    def lookup_robot_in_gridmap_frame(self, gridmap_frame: str):
        # point at base origin expressed in base_footprint
        p = PointStamped()
        p.header.frame_id = self.base_frame
        p.header.stamp = rclpy.time.Time().to_msg()  # latest
        p.point.x = 0.0
        p.point.y = 0.0
        p.point.z = 0.0

        try:
            tf = self.tf_buffer.lookup_transform(gridmap_frame, self.base_frame, rclpy.time.Time())
            p_out = do_transform_point(p, tf)
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed ({gridmap_frame} <- {self.base_frame}): {e}")
            return None

        rx = p_out.point.x
        ry = p_out.point.y

        # yaw from rotation (still ok)
        yaw = quat_to_yaw(tf.transform.rotation)
        return rx, ry, yaw


    def world_to_map_indices(self, x_w, y_w, info, rows, cols):
        """
        Convert world (gridmap frame) -> (row, col) indices into the matrix.
        Handles map pose (translation + yaw).
        Assumes map is axis-aligned in its pose yaw (no roll/pitch).
        """
        res = float(info.resolution)
        cx = float(info.pose.position.x)
        cy = float(info.pose.position.y)
        q = info.pose.orientation

        self.get_logger().info(f"gridmap pose yaw = {quat_to_yaw(q)}")
        yaw = quat_to_yaw(q)

        # Transform world point into map-local coordinates (centered at map center)
        dx = x_w - cx
        dy = y_w - cy

        c = math.cos(-yaw)
        s = math.sin(-yaw)
        x_m = c * dx - s * dy
        y_m = s * dx + c * dy

        # Map extents (meters)
        lx = float(info.length_x)
        ly = float(info.length_y)

        # Convert to indices (col along x, row along y)
        # x in [-lx/2, lx/2)
        # y in [-ly/2, ly/2)
        # col = int(math.floor((x_m + lx / 2.0) / res)) # x -> col
        # row = int(math.floor((y_m + ly / 2.0) / res)) # y-> row

        # row = int(math.floor((x_m + lx / 2.0) / res)) # x -> row
        # col = int(math.floor((y_m + ly / 2.0) / res)) # y -> col
        row = int(math.floor((lx / 2.0 - x_m) / res))  # x -> row, flipped
        col = int(math.floor((ly / 2.0 - y_m) / res))  # y -> col, flipped


        # Clamp
        row = max(0, min(rows - 1, row))
        col = max(0, min(cols - 1, col))
        return row, col

    def crop_window(self, mat, center_rc, win_rows, win_cols):
        """
        Crop with padding as NaN if near borders.
        """
        rows, cols = mat.shape
        cr, cc = center_rc

        r0 = cr - win_rows // 2
        c0 = cc - win_cols // 2

        out = np.full((win_rows, win_cols), np.nan, dtype=np.float32)

        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(rows, r0 + win_rows)
        cc1 = min(cols, c0 + win_cols)

        out_r0 = rr0 - r0
        out_c0 = cc0 - c0
        out_r1 = out_r0 + (rr1 - rr0)
        out_c1 = out_c0 + (cc1 - cc0)

        out[out_r0:out_r1, out_c0:out_c1] = mat[rr0:rr1, cc0:cc1]
        return out

    def compute_slope_mag(self, elev: np.ndarray, res: float) -> np.ndarray:
        """
        Slope magnitude proxy: ||grad(h)||, using central differences.
        Returns NaN where elev is NaN.
        """
        # Use np.gradient (handles edges with 1st order differences)
        # gradient returns (d/drow, d/dcol); row is y, col is x
        dy, dx = np.gradient(elev.astype(np.float32), res, res)
        slope = np.sqrt(dx * dx + dy * dy)
        slope[~np.isfinite(elev)] = np.nan
        return slope.astype(np.float32)

    def compute_roughness(self, elev: np.ndarray, res: float) -> np.ndarray:
        """
        Roughness proxy: mean absolute deviation from local mean in a square window.
        Fast enough for local map sizes.
        """
        if not self.use_roughness:
            return np.zeros_like(elev, dtype=np.float32)

        rad_cells = max(1, int(round(self.rough_radius / res)))
        k = 2 * rad_cells + 1

        # Simple box filter via integral image (ignoring NaNs by masking)
        mask = np.isfinite(elev).astype(np.float32)
        elev0 = np.nan_to_num(elev, nan=0.0).astype(np.float32)

        # integral images
        S = elev0.cumsum(axis=0).cumsum(axis=1)
        M = mask.cumsum(axis=0).cumsum(axis=1)

        def box_sum(ii, r0, c0, r1, c1):
            # inclusive-exclusive sum over [r0,r1) x [c0,c1)
            A = ii[r1 - 1, c1 - 1]
            B = ii[r0 - 1, c1 - 1] if r0 > 0 else 0.0
            C = ii[r1 - 1, c0 - 1] if c0 > 0 else 0.0
            D = ii[r0 - 1, c0 - 1] if (r0 > 0 and c0 > 0) else 0.0
            return A - B - C + D

        rows, cols = elev.shape
        mean = np.full_like(elev0, np.nan, dtype=np.float32)

        for r in range(rows):
            r0 = max(0, r - rad_cells)
            r1 = min(rows, r + rad_cells + 1)
            for c in range(cols):
                c0 = max(0, c - rad_cells)
                c1 = min(cols, c + rad_cells + 1)
                m = box_sum(M, r0, c0, r1, c1)
                if m <= 1e-3:
                    continue
                s = box_sum(S, r0, c0, r1, c1)
                mean[r, c] = s / m

        rough = np.abs(elev0 - mean)
        rough[mask < 0.5] = np.nan
        return rough.astype(np.float32)

    def apply_axis_debug(self, mat: np.ndarray) -> np.ndarray:
        out = mat
        if self.swap_xy:
            out = out.T
        if self.flip_x:
            # x corresponds to columns
            out = np.flip(out, axis=1)
        if self.flip_y:
            # y corresponds to rows
            out = np.flip(out, axis=0)
        return out

    def on_map(self, msg: GridMap):
        self.last_msg = msg

    def on_timer(self):
        if self.last_msg is None:
            return

        msg = self.last_msg

        if self.elevation_layer not in msg.layers:
            self.get_logger().warn(
                f"Incoming GridMap missing elevation_layer='{self.elevation_layer}'. "
                f"Available: {list(msg.layers)}"
            )
            return

        gridmap_frame = msg.header.frame_id if msg.header.frame_id else "terrain_map"
        robot = self.lookup_robot_in_gridmap_frame(gridmap_frame)#, msg.header.stamp)

        if robot is None:
            return
        rx, ry, _ = robot

        # Determine grid shape from message info
        # Rows/cols are inferred from the data layer layout
        elev_layer_idx = msg.layers.index(self.elevation_layer)
        elev_msg = msg.data[elev_layer_idx]

        # Infer rows/cols from layout dims if present; else from length/resolution
        if len(elev_msg.layout.dim) >= 2:
            cols = int(elev_msg.layout.dim[0].size)
            rows = int(elev_msg.layout.dim[1].size)
        else:
            # fallback
            res = float(msg.info.resolution)
            rows = int(round(float(msg.info.length_x) / res))
            cols = int(round(float(msg.info.length_y) / res))

        self.get_logger().info(f"robot in {gridmap_frame}: rx={rx:.3f}, ry={ry:.3f}")


        elev_global = parse_layer_to_mat(elev_msg, rows, cols)
        # elev_global = self.apply_axis_debug(elev_global)

        res = float(msg.info.resolution)

        # Local window in cells
        win_cols = max(3, int(round(self.local_size_x / res)))
        win_rows = max(3, int(round(self.local_size_y / res)))
        # enforce odd sizes to center nicely
        if win_cols % 2 == 0:
            win_cols += 1
        if win_rows % 2 == 0:
            win_rows += 1

        # Center index (row,col) for robot in global map
        center_rc = self.world_to_map_indices(rx, ry, msg.info, elev_global.shape[0], elev_global.shape[1])
        self.get_logger().info(
            f"rx={rx:.3f}, ry={ry:.3f} -> center_rc(row=x, col=y)=({center_rc[0]}, {center_rc[1]})"
        )


        # elev_local = self.crop_window(elev_global, center_rc, win_rows, win_cols)

        # # Compute cost_base
        # slope = self.compute_slope_mag(elev_local, res)
        # if self.use_roughness:
        #     rough = self.compute_roughness(elev_local, res)
        # else:
        #     rough = np.zeros_like(elev_local, dtype=np.float32)

        # slope_n = np.clip(slope / max(1e-6, self.slope_max), 0.0, 1.0)
        # if self.use_roughness:
        #     rough_n = np.clip(rough / max(1e-6, self.roughness_max), 0.0, 1.0)
        # else:
        #     rough_n = 0.0

        # cost = self.w_slope * slope_n + self.w_rough * rough_n
        # cost[~np.isfinite(elev_local)] = np.nan
        # cost = np.clip(cost, 0.0, 1.0).astype(np.float32)

        # # Build output GridMap
        # out = GridMap()
        # out.header.stamp = self.get_clock().now().to_msg()

        # # Publish in base frame (centered at robot)
        # if self.publish_in_base_frame:
        #     out.header.frame_id = self.base_frame
        #     out.info.pose.position.x = 0.0
        #     out.info.pose.position.y = 0.0
        #     out.info.pose.position.z = 0.0
        #     out.info.pose.orientation.w = 1.0
        # else:
        #     out.header.frame_id = gridmap_frame
        #     out.info.pose.position.x = float(rx)
        #     out.info.pose.position.y = float(ry)
        #     out.info.pose.position.z = 0.0
        #     out.info.pose.orientation.w = 1.0

        # out.info.resolution = res
        # out.info.length_x = float(win_cols) * res
        # out.info.length_y = float(win_rows) * res

        # out.layers = ["elevation_local", "cost_base"]
        # out.basic_layers = ["elevation_local"]

        # # Use incoming layout (the exact way RViz expect it)
        # layout = msg.data[elev_layer_idx].layout

        # # Make a deep-ish copy and edit dims
        # layout_out = Float32MultiArray().layout
        # layout_out.dim = [MultiArrayDimension(), MultiArrayDimension()]
        # layout_out.data_offset = 0

        # # grid_map expects dim[0]=cols, dim[1]=rows with Eigen column-major
        # layout_out.dim[0].label = layout.dim[0].label
        # layout_out.dim[1].label = layout.dim[1].label

        # rows, cols = elev_local.shape
        # layout_out.dim[0].size = cols
        # layout_out.dim[1].size = rows
        # layout_out.dim[0].stride = cols * rows
        # layout_out.dim[1].stride = rows


        # elev_out = Float32MultiArray()
        # elev_out.layout = layout_out
        # elev_out.data = elev_local.astype(np.float32).ravel(order="F").tolist()

        # cost_out = Float32MultiArray()
        # cost_out.layout = layout_out
        # cost_out.data = cost.astype(np.float32).ravel(order="F").tolist()

        # out.data = [elev_out, cost_out]


        # # out.data = [
        # #     make_float32_multiarray(elev_local, "row_index", "col_index"),
        # #     make_float32_multiarray(cost, "row_index", "col_index"),
        # # ]

        # out.outer_start_index = 0
        # out.inner_start_index = 0

        # out.info.pose.orientation.x = 0.0
        # out.info.pose.orientation.y = 0.0
        # out.info.pose.orientation.z = 0.0
        # out.info.pose.orientation.w = 1.0


        # self.pub.publish(out)


        # - - - - - - - - - - - - - - - - - -
        # same r0/c0 math as crop_window
        cr, cc = center_rc
        r0 = cr - win_rows // 2
        c0 = cc - win_cols // 2

        rr0 = max(0, r0)
        cc0 = max(0, c0)
        rr1 = min(rows, r0 + win_rows)
        cc1 = min(cols, c0 + win_cols)

        # Start with full-size NaN maps
        elev_local_full = np.full_like(elev_global, np.nan, dtype=np.float32)

        # extract the elevation window from the global map
        elev_window = elev_global[rr0:rr1, cc0:cc1]

        # place that window back into the full map
        elev_local_full[rr0:rr1, cc0:cc1] = elev_window

        z_offset = 0.02
        elev_local_vis = elev_local_full.copy()
        mask = np.isfinite(elev_local_vis)
        elev_local_vis[mask] += z_offset


        # compute slope/cost **on the window**, then embed back too
        slope_window = self.compute_slope_mag(elev_window, res)
        if self.use_roughness:
            rough_window = self.compute_roughness(elev_window, res)
        else:
            rough_window = np.zeros_like(elev_window, dtype=np.float32)

        slope_n_window = np.clip(slope_window / max(1e-6, self.slope_max), 0.0, 1.0)
        if self.use_roughness:
            rough_n_window = np.clip(rough_window / max(1e-6, self.roughness_max), 0.0, 1.0)
        else:
            rough_n_window = 0.0

        cost_window = self.w_slope * slope_n_window + self.w_rough * rough_n_window
        cost_window[~np.isfinite(elev_window)] = np.nan
        cost_window = np.clip(cost_window, 0.0, 1.0).astype(np.float32)

        cost_full = np.full_like(elev_global, np.nan, dtype=np.float32)
        cost_full[rr0:rr1, cc0:cc1] = cost_window

        out = GridMap()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = msg.header.frame_id   # "terrain_map"

        # Copy global map geometry so it sits exactly on top
        out.info = msg.info

        out.layers = ["elevation_local", "cost_base"]
        out.basic_layers = ["elevation_local"]

        rows, cols = elev_local_full.shape

        layout_out = Float32MultiArray().layout
        layout_out.dim = [MultiArrayDimension(), MultiArrayDimension()]
        layout_out.data_offset = 0

        # grid_map expects dim[0]=cols, dim[1]=rows with column-major
        layout_out.dim[0].label = "column_index"
        layout_out.dim[1].label = "row_index"
        layout_out.dim[0].size = cols
        layout_out.dim[1].size = rows
        layout_out.dim[0].stride = cols * rows
        layout_out.dim[1].stride = rows

        elev_out = Float32MultiArray()
        elev_out.layout = layout_out
        elev_out.data = elev_local_vis.astype(np.float32).ravel(order="F").tolist()

        cost_out = Float32MultiArray()
        cost_out.layout = layout_out
        cost_out.data = cost_full.astype(np.float32).ravel(order="F").tolist()

        out.data = [elev_out, cost_out]
        out.outer_start_index = 0
        out.inner_start_index = 0

        self.pub.publish(out)




def main():
    rclpy.init()
    node = LocalTraversabilityCost()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
