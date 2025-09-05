#!/usr/bin/env python3
"""
Policy iteration path planning on a grayscale heightmap.

- White = high, black = low
- Blocks cells above a slope threshold
- Cost = distance + uphill/downhill + slope + roughness
- Computes optimal policy for entire grid toward a goal
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import uniform_filter
import csv

# --------------------------- Policy Iteration ---------------------------

def compute_fields(img_path, scale_to, cell_size_m, height_scale_m_per_intensity, rough_win):
    img = Image.open(img_path).convert("L")
    if scale_to is not None:
        img = img.resize((scale_to, scale_to), Image.Resampling.BILINEAR)
    Z = np.asarray(img, dtype=np.float32) / 255.0         # normalized [0,1]
    H = Z * float(height_scale_m_per_intensity)           # elevation [m]

    dHy, dHx = np.gradient(H, float(cell_size_m))
    slope_rad = np.arctan(np.sqrt(dHx**2 + dHy**2))
    slope_deg = np.degrees(slope_rad)

    mean_H = uniform_filter(H, size=rough_win)
    mean_H2 = uniform_filter(H**2, size=rough_win)
    rough = np.sqrt(np.maximum(mean_H2 - mean_H**2, 0.0))

    return Z, H, slope_deg, rough

def policy_iteration(Z, H, slope_deg, rough, goal_rc,
                    cell_size_m, max_slope_deg, w_up, w_down, w_slope, w_rough,
                    max_iterations=100, tolerance=1e-6, gamma=0.99):
    """
    Policy iteration for path planning.
    
    Returns:
        policy: (h, w) array of action indices for each cell
        value: (h, w) array of values (negative costs-to-go)
        actions: list of (dy, dx, multiplier) tuples
    """
    h, w = Z.shape
    traversable = slope_deg <= max_slope_deg
    
    # 8-connected actions (same as A*)
    actions = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
               (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
               (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]
    n_actions = len(actions)
    
    def in_bounds(y, x): 
        return 0 <= y < h and 0 <= x < w
    
    def step_cost(y0, x0, y1, x1, mult):
        """Cost of moving from (y0,x0) to (y1,x1)"""
        if not (in_bounds(y1, x1) and traversable[y1, x1]):
            return np.inf
        
        base = mult * cell_size_m
        dz = H[y1, x1] - H[y0, x0]
        uphill   = max(0.0,  dz)
        downhill = max(0.0, -dz)
        s_deg = slope_deg[y1, x1]
        r = rough[y1, x1]
        return base + w_up*uphill + w_down*downhill + w_slope*(s_deg/max_slope_deg) + w_rough*r
    
    # Initialize
    gy, gx = goal_rc
    if not (in_bounds(gy, gx) and traversable[gy, gx]):
        raise ValueError("Goal is not traversable at this slope limit.")
    
    # Value function: V(s) represents negative cost-to-go (higher = better)
    # We use negative costs so that value iteration maximizes value
    V = np.full((h, w), -np.inf, dtype=np.float64)
    V[gy, gx] = 0.0  # Goal has zero cost
    
    # Initialize random policy (avoid goal cell)
    policy = np.random.randint(0, n_actions, (h, w))
    
    print("Starting policy iteration...")
    
    for iteration in range(max_iterations):
        # Policy Evaluation: solve V^π = R + γ P^π V^π
        # We'll use iterative policy evaluation
        V_old = V.copy()
        
        # Iterative policy evaluation
        for _ in range(50):  # Inner iterations for policy evaluation
            V_new = V.copy()
            
            for y in range(h):
                for x in range(w):
                    if (y, x) == (gy, gx):
                        continue  # Goal state
                    if not traversable[y, x]:
                        V_new[y, x] = -np.inf
                        continue
                    
                    # Follow current policy
                    action_idx = policy[y, x]
                    dy, dx, mult = actions[action_idx]
                    ny, nx = y + dy, x + dx
                    
                    cost = step_cost(y, x, ny, nx, mult)
                    if np.isfinite(cost):
                        V_new[y, x] = -cost + gamma * V[ny, nx]
                    else:
                        V_new[y, x] = -np.inf
            
            if np.allclose(V_new, V, rtol=tolerance, atol=tolerance):
                break
            V = V_new
        
        # Policy Improvement
        policy_stable = True
        new_policy = policy.copy()
        
        for y in range(h):
            for x in range(w):
                if (y, x) == (gy, gx) or not traversable[y, x]:
                    continue
                
                # Find best action
                best_value = -np.inf
                best_action = policy[y, x]
                
                for action_idx, (dy, dx, mult) in enumerate(actions):
                    ny, nx = y + dy, x + dx
                    cost = step_cost(y, x, ny, nx, mult)
                    
                    if np.isfinite(cost):
                        value = -cost + gamma * V[ny, nx]
                        if value > best_value:
                            best_value = value
                            best_action = action_idx
                
                new_policy[y, x] = best_action
                if new_policy[y, x] != policy[y, x]:
                    policy_stable = False
        
        policy = new_policy
        
        # Check convergence
        value_change = np.max(np.abs(V - V_old))
        print(f"Iteration {iteration + 1}: Value change = {value_change:.6f}")
        
        if policy_stable:
            print(f"Policy converged after {iteration + 1} iterations!")
            break
        
        if value_change < tolerance:
            print(f"Value function converged after {iteration + 1} iterations!")
            break
    
    return policy, V, actions

def extract_path_from_policy(policy, actions, start_rc, goal_rc, max_steps=1000):
    """Extract path by following the policy from start to goal."""
    path = []
    y, x = start_rc
    gy, gx = goal_rc
    
    for step in range(max_steps):
        path.append((y, x))
        
        if (y, x) == (gy, gx):
            return path
        
        # Follow policy
        action_idx = policy[y, x]
        dy, dx, _ = actions[action_idx]
        y, x = y + dy, x + dx
        
        # Check bounds
        h, w = policy.shape
        if not (0 <= y < h and 0 <= x < w):
            print(f"Path went out of bounds at step {step}")
            break
    
    print(f"Path did not reach goal within {max_steps} steps")
    return path

# --------------------------- CLI + UI (adapted from original) ---------------------------

def pick_points_interactive(Z, title_extra=""):
    """Click to choose start+goal. Left click twice, then close the window."""
    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(Z)
    ax.set_title(f"Left-click START then END. Close window when done.\n{title_extra}")
    ax.axis("off")
    pts = plt.ginput(2, timeout=0)
    plt.close(fig)
    if len(pts) != 2:
        raise RuntimeError("Did not receive two clicks.")
    (x0, y0), (x1, y1) = pts
    start = (int(round(y0)), int(round(x0)))
    goal  = (int(round(y1)), int(round(x1)))
    return start, goal

def draw_policy_and_path(Z, policy, actions, start_rc, goal_rc, path, value, 
                        out_path=None, title_extra=""):
    """Draw the policy field and the resulting path."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Left plot: Policy field
    h, w = Z.shape
    Y, X = np.meshgrid(range(w), range(h))
    
    # Create arrow components from policy
    U = np.zeros_like(Y, dtype=float)
    V = np.zeros_like(X, dtype=float)
    
    for y in range(h):
        for x in range(w):
            if np.isfinite(value[y, x]) and value[y, x] > -np.inf:
                dy, dx, _ = actions[policy[y, x]]
                U[y, x] = dx * 0.3  # Scale arrows
                V[y, x] = -dy * 0.3  # Flip y for display
    
    ax1.imshow(Z, alpha=0.7)
    ax1.quiver(X, Y, U, V, scale=1, scale_units='xy', angles='xy', width=0.002)
    ax1.scatter([goal_rc[1]], [goal_rc[0]], c='red', s=100, marker='*', label='Goal')
    ax1.set_title(f"Policy Field\n{title_extra}")
    ax1.axis("off")
    ax1.legend()
    
    # Right plot: Path
    ax2.imshow(Z)
    ax2.scatter([start_rc[1], goal_rc[1]], [start_rc[0], goal_rc[0]], 
               c=['green', 'red'], s=100, marker='*', label=['Start', 'Goal'])
    
    if path:
        ys, xs = zip(*path)
        ax2.plot(xs, ys, 'b-', linewidth=2, label='Path')
        cost = len(path) - 1  # Simple path length cost
        ax2.set_title(f"Path: {len(path)} steps")
    else:
        ax2.text(Z.shape[1]*0.5, Z.shape[0]*0.5, "No path found", 
                ha='center', va='center', fontsize=14, color='red')
        ax2.set_title("No path found")
    
    ax2.axis("off")
    ax2.legend()
    
    plt.tight_layout()
    
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved figure to: {out_path}")
    plt.show()

def parse_tuple(s):
    r, c = s.split(",")
    return (int(r), int(c))

def parse_wh_pair(s):
    a, b = s.split(",")
    return float(a), float(b)

def pixels_to_world_polyline(path_rc, grid_w, grid_h, world_w, world_h):
    """Convert (row,col) pixel centers to (x,y) in meters."""
    sx = world_w / grid_w
    sy = world_h / grid_h
    pts = []
    for r, c in path_rc:
        x = (c + 0.5) * sx
        y_img_down = (r + 0.5) * sy
        y = world_h - y_img_down
        pts.append((x, y))
    return np.asarray(pts, dtype=np.float64)

def resample_polyline_m(pts_xy, ds):
    """Evenly resample polyline at spacing ds (meters)."""
    if len(pts_xy) < 2:
        return pts_xy
    dseg = np.linalg.norm(np.diff(pts_xy, axis=0), axis=1)
    cum = np.concatenate(([0.0], np.cumsum(dseg)))
    L = cum[-1]
    if L == 0:
        return pts_xy[:1]
    s_samples = np.arange(0.0, L + 1e-9, ds)
    out = []
    j = 0
    for s in s_samples:
        while j+1 < len(cum) and cum[j+1] < s:
            j += 1
        if j+1 >= len(cum):
            out.append(pts_xy[-1])
            continue
        t = (s - cum[j]) / (cum[j+1] - cum[j] + 1e-12)
        p = (1 - t) * pts_xy[j] + t * pts_xy[j+1]
        out.append(p)
    return np.asarray(out)

def heading_yaw(pts_xy):
    """Compute yaw (rad) from segment directions."""
    if len(pts_xy) == 1:
        return np.array([0.0])
    v = np.diff(pts_xy, axis=0)
    yaw = np.arctan2(v[:,1], v[:,0])
    yaw = np.concatenate([yaw, yaw[-1:]])
    return yaw

def main():
    p = argparse.ArgumentParser(description="Policy iteration on grayscale heightmap")
    p.add_argument("--image", required=True, help="Path to grayscale heightmap image")
    p.add_argument("--scale", type=int, default=64, help="Downsample size (NxN) - smaller for policy iteration")
    p.add_argument("--cell-size", type=float, default=1.0, help="Meters per pixel")
    p.add_argument("--height-scale", type=float, default=30.0,
                   help="Meters per 0→1 intensity change")
    p.add_argument("--rough-win", type=int, default=5, help="Roughness window size")
    p.add_argument("--max-slope", type=float, default=20.0, help="Max slope (deg)")
    p.add_argument("--w-up", type=float, default=4.0, help="Uphill cost weight")
    p.add_argument("--w-down", type=float, default=1.0, help="Downhill cost weight")
    p.add_argument("--w-slope", type=float, default=2.0, help="Slope penalty weight")
    p.add_argument("--w-rough", type=float, default=0.5, help="Roughness penalty weight")
    p.add_argument("--start", type=parse_tuple, help="Start 'row,col' on downsampled grid")
    p.add_argument("--goal",  type=parse_tuple, help="Goal  'row,col' on downsampled grid")
    p.add_argument("--save",  type=str, default=None, help="Path to save the result image")
    p.add_argument("--no-click", action="store_true", help="Disable click-picking")
    p.add_argument("--world-size", type=str, default="40,40",
               help="World size in meters as 'width,height'")
    p.add_argument("--sample-ds", type=float, default=0.5,
                help="Waypoint spacing in meters")
    p.add_argument("--export-csv", type=str, default=None,
                help="Path to save waypoints CSV")
    p.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    p.add_argument("--max-iter", type=int, default=100, help="Max policy iterations")
    args = p.parse_args()

    Z, H, slope_deg, rough = compute_fields(
        args.image, args.scale, args.cell_size, args.height_scale, args.rough_win
    )

    start_rc, goal_rc = args.start, args.goal
    if (start_rc is None or goal_rc is None) and not args.no_click:
        print("Click to choose START and END (close window after two clicks)...")
        start_rc, goal_rc = pick_points_interactive(Z, "Policy Iteration Planner")
    
    if start_rc is None or goal_rc is None:
        h, w = Z.shape
        start_rc = start_rc or (10, 10)
        goal_rc  = goal_rc  or (h-11, w-11)

    try:
        print(f"Computing policy for goal at {goal_rc}...")
        policy, value, actions = policy_iteration(
            Z, H, slope_deg, rough, goal_rc,
            args.cell_size, args.max_slope, args.w_up, args.w_down, 
            args.w_slope, args.w_rough, args.max_iter, gamma=args.gamma
        )
        
        print(f"Extracting path from {start_rc} to {goal_rc}...")
        path = extract_path_from_policy(policy, actions, start_rc, goal_rc)
        
    except ValueError as e:
        print(f"Planner error: {e}")
        policy, value, actions, path = None, None, None, None

    if policy is not None:
        title = (f"γ={args.gamma}, max_slope={args.max_slope}°, weights:"
                f" up={args.w_up}, down={args.w_down}, slope={args.w_slope}, rough={args.w_rough}")
        draw_policy_and_path(Z, policy, actions, start_rc, goal_rc, path, value,
                            out_path=args.save, title_extra=title)

        # CSV export (same as original)
        if path and args.export_csv:
            grid_h, grid_w = Z.shape
            world_w_m, world_h_m = parse_wh_pair(args.world_size)

            pts_xy = pixels_to_world_polyline(path, grid_w, grid_h, world_w_m, world_h_m)
            
            sx = world_w_m / grid_w
            sy = world_h_m / grid_h
            zs = []
            for (x, y) in pts_xy:
                c = int(np.clip(x / sx - 0.5, 0, grid_w - 1))
                r = int(np.clip((world_h_m - y) / sy - 0.5, 0, grid_h - 1))
                z = (Z[r, c] * args.height_scale)
                zs.append(z)
            zs = np.asarray(zs)

            pts_xy_rs = resample_polyline_m(pts_xy, args.sample_ds)
            zs_rs = []
            for (x, y) in pts_xy_rs:
                c = int(np.clip(x / sx - 0.5, 0, grid_w - 1))
                r = int(np.clip((world_h_m - y) / sy - 0.5, 0, grid_h - 1))
                zs_rs.append(Z[r, c] * args.height_scale)
            zs_rs = np.asarray(zs_rs)

            yaw_rs = heading_yaw(pts_xy_rs)

            with open(args.export_csv, "w", newline="") as f:
                import csv
                wcsv = csv.writer(f)
                wcsv.writerow(["x_m", "y_m", "z_m", "yaw_rad"])
                for (x, y), z, th in zip(pts_xy_rs, zs_rs, yaw_rs):
                    wcsv.writerow([f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", f"{th:.6f}"])
            print(f"Exported {len(pts_xy_rs)} waypoints to {args.export_csv}")

if __name__ == "__main__":
    main()