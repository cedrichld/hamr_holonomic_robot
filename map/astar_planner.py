#!/usr/bin/env python3
"""
A* path planning on a grayscale heightmap.

- White = high, black = low
- Blocks cells above a slope threshold
- Cost = distance + uphill/downhill + slope + roughness
- Pick start/end by clicking (or pass via CLI)
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from heapq import heappush, heappop
from scipy.ndimage import uniform_filter
import csv

# --------------------------- Core planner ---------------------------

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

def astar(Z, H, slope_deg, rough, start_rc, goal_rc,
          cell_size_m, max_slope_deg, w_up, w_down, w_slope, w_rough):
    h, w = Z.shape
    traversable = slope_deg <= max_slope_deg

    # 8-connected motion
    moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, np.sqrt(2)), (-1, 1, np.sqrt(2)),
             (1, -1, np.sqrt(2)), (1, 1, np.sqrt(2))]

    def in_bounds(y, x): return 0 <= y < h and 0 <= x < w

    def step_cost(y0, x0, y1, x1, mult):
        if not traversable[y1, x1]:
            return np.inf
        base = mult * cell_size_m
        dz = H[y1, x1] - H[y0, x0]
        uphill   = max(0.0,  dz)
        downhill = max(0.0, -dz)
        s_deg = slope_deg[y1, x1]
        r     = rough[y1, x1]
        return base + w_up*uphill + w_down*downhill + w_slope*(s_deg/max_slope_deg) + w_rough*r

    def heuristic(y, x, gy, gx):
        return np.hypot(gy - y, gx - x) * cell_size_m

    sy, sx = start_rc
    gy, gx = goal_rc
    if not (in_bounds(sy, sx) and in_bounds(gy, gx)):
        raise ValueError("Start or goal is out of bounds.")
    if not (traversable[sy, sx] and traversable[gy, gx]):
        raise ValueError("Start or goal is not traversable at this slope limit.")

    g = np.full((h, w), np.inf, dtype=np.float64)
    f = np.full((h, w), np.inf, dtype=np.float64)
    py = np.full((h, w), -1, dtype=np.int32)
    px = np.full((h, w), -1, dtype=np.int32)

    g[sy, sx] = 0.0
    f[sy, sx] = heuristic(sy, sx, gy, gx)
    pq, visited = [], np.zeros((h, w), dtype=bool)
    heappush(pq, (f[sy, sx], (sy, sx)))

    while pq:
        _, (y, x) = heappop(pq)
        if visited[y, x]: 
            continue
        visited[y, x] = True

        if (y, x) == (gy, gx):
            path = [(y, x)]
            while not (y == sy and x == sx):
                y, x = py[y, x], px[y, x]
                path.append((y, x))
            path.reverse()
            return path, g[gy, gx], traversable

        for dy, dx, mult in moves:
            ny, nx = y + dy, x + dx
            if not in_bounds(ny, nx) or visited[ny, nx]:
                continue
            c = step_cost(y, x, ny, nx, mult)
            if not np.isfinite(c):
                continue
            cand = g[y, x] + c
            if cand < g[ny, nx]:
                g[ny, nx] = cand
                f[ny, nx] = cand + heuristic(ny, nx, gy, gx)
                py[ny, nx] = y
                px[ny, nx] = x
                heappush(pq, (f[ny, nx], (ny, nx)))

    return None, np.inf, traversable

# --------------------------- CLI + UI ---------------------------

def pick_points_interactive(Z):
    """Click to choose start+goal. Left click twice, then close the window."""
    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(Z)
    ax.set_title("Left-click START then END. Close window when done.")
    ax.axis("off")
    pts = plt.ginput(2, timeout=0)  # infinite timeout; close window to finish
    plt.close(fig)
    if len(pts) != 2:
        raise RuntimeError("Did not receive two clicks.")
    (x0, y0), (x1, y1) = pts
    start = (int(round(y0)), int(round(x0)))
    goal  = (int(round(y1)), int(round(x1)))
    return start, goal

def draw_result(Z, start_rc, goal_rc, path, cost, out_path=None, title_extra=""):
    fig, ax = plt.subplots(figsize=(7,7))
    ax.imshow(Z)
    t = f"A* path  |  start={start_rc}, goal={goal_rc}  |  cost={cost:.1f}"
    if title_extra:
        t += f"\n{title_extra}"
    ax.set_title(t)
    ax.axis("off")
    if path:
        ys, xs = zip(*path)
        ax.plot(xs, ys, linewidth=2)
        ax.scatter([start_rc[1], goal_rc[1]], [start_rc[0], goal_rc[0]], s=40)
    else:
        ax.text(Z.shape[1]*0.5, Z.shape[0]*0.5, "No path found", ha='center', va='center')
    if out_path:
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        print(f"Saved figure to: {out_path}")
    plt.show()

def parse_tuple(s):
    # format: "row,col"
    r, c = s.split(",")
    return (int(r), int(c))

def parse_wh_pair(s):
    a, b = s.split(",")
    return float(a), float(b)

def pixels_to_world_polyline(path_rc, grid_w, grid_h, world_w, world_h):
    """Convert (row,col) pixel centers to (x,y) in meters.
       x right, y up; flips the image's y-down to y-up."""
    sx = world_w / grid_w
    sy = world_h / grid_h
    pts = []
    for r, c in path_rc:
        x = (c + 0.5) * sx
        y_img_down = (r + 0.5) * sy
        y = world_h - y_img_down   # flip to y-up
        pts.append((x, y))
    return np.asarray(pts, dtype=np.float64)

def add_z_from_height(pts_xy, Z_norm, height_scale_m):
    """Sample elevation z from normalized grayscale (nearest cell)."""
    h, w = Z_norm.shape
    world_w = 1.0  # we will scale indices separately below
    # Map back to pixel index for sampling
    zs = []
    for x, y in pts_xy:
        # We'll temporarily assume world size is (W,H) later; for now zs placeholder
        zs.append(0.0)
    return np.asarray(zs, dtype=np.float64)

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
    """Compute yaw (rad) from segment directions; duplicate last heading for final point."""
    if len(pts_xy) == 1:
        return np.array([0.0])
    v = np.diff(pts_xy, axis=0)
    yaw = np.arctan2(v[:,1], v[:,0])
    yaw = np.concatenate([yaw, yaw[-1:]])
    return yaw

def main():
    p = argparse.ArgumentParser(description="A* on grayscale heightmap")
    p.add_argument("--image", required=True, help="Path to grayscale heightmap image")
    p.add_argument("--scale", type=int, default=256, help="Downsample size (NxN)")
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
               help="World size in meters as 'width,height' (default 40,40)")
    p.add_argument("--sample-ds", type=float, default=0.5,
                help="Waypoint spacing in meters (default 0.5)")
    p.add_argument("--export-csv", type=str, default=None,
                help="Path to save waypoints CSV")
    args = p.parse_args()

    Z, H, slope_deg, rough = compute_fields(
        args.image, args.scale, args.cell_size, args.height_scale, args.rough_win
    )

    start_rc, goal_rc = args.start, args.goal
    if (start_rc is None or goal_rc is None) and not args.no_click:
        print("Click to choose START and END (close window after two clicks)...")
        start_rc, goal_rc = pick_points_interactive(Z)
    if start_rc is None or goal_rc is None:
        # fallback defaults
        h, w = Z.shape
        start_rc = start_rc or (10, 10)
        goal_rc  = goal_rc  or (h-11, w-11)

    try:
        path, cost, trav = astar(
            Z, H, slope_deg, rough, start_rc, goal_rc,
            args.cell_size, args.max_slope, args.w_up, args.w_down, args.w_slope, args.w_rough
        )
    except ValueError as e:
        print(f"Planner error: {e}")
        path, cost = None, np.inf

    title = (f"max_slope={args.max_slope}°, weights:"
             f" up={args.w_up}, down={args.w_down}, slope={args.w_slope}, rough={args.w_rough}")
    draw_result(Z, start_rc, goal_rc, path, cost, out_path=args.save, title_extra=title)

    # --- Waypoint generation & CSV export ---
    if path and args.export_csv:
        grid_h, grid_w = Z.shape
        world_w_m, world_h_m = parse_wh_pair(args.world_size)

        # (row,col) -> (x,y) meters (y-up)
        pts_xy = pixels_to_world_polyline(path, grid_w, grid_h, world_w_m, world_h_m)

        # add elevation z from image (use normalized Z * height_scale)
        # map (x,y) -> nearest pixel for z sampling
        sx = world_w_m / grid_w
        sy = world_h_m / grid_h
        zs = []
        for (x, y) in pts_xy:
            c = int(np.clip(x / sx - 0.5, 0, grid_w - 1))
            r = int(np.clip((world_h_m - y) / sy - 0.5, 0, grid_h - 1))
            z = (Z[r, c] * args.height_scale)  # meters
            zs.append(z)
        zs = np.asarray(zs)

        # resample to evenly spaced waypoints
        pts_xy_rs = resample_polyline_m(pts_xy, args.sample_ds)
        # recompute z at resampled points
        zs_rs = []
        for (x, y) in pts_xy_rs:
            c = int(np.clip(x / sx - 0.5, 0, grid_w - 1))
            r = int(np.clip((world_h_m - y) / sy - 0.5, 0, grid_h - 1))
            zs_rs.append(Z[r, c] * args.height_scale)
        zs_rs = np.asarray(zs_rs)

        # yaw from the resampled path
        yaw_rs = heading_yaw(pts_xy_rs)

        # write CSV: x,y,z,yaw
        with open(args.export_csv, "w", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow(["x_m", "y_m", "z_m", "yaw_rad"])
            for (x, y), z, th in zip(pts_xy_rs, zs_rs, yaw_rs):
                wcsv.writerow([f"{x:.3f}", f"{y:.3f}", f"{z:.3f}", f"{th:.6f}"])
        print(f"Exported {len(pts_xy_rs)} waypoints to {args.export_csv}")

if __name__ == "__main__":
    main()
