#!/usr/bin/env python3
import numpy as np
from PIL import Image

def load_heightmap_png(path: str, z_max_m: float = 2.5) -> np.ndarray:
    """
    Loads a PNG heightmap, converts to grayscale, maps [0..255] -> [0..z_max_m] meters linearly.
    Returns H in meters, shape (N,N).
    """
    img = Image.open(path).convert("L")  # grayscale 8-bit
    I = np.asarray(img, dtype=np.float64)  # 0..255
    H = (I / 255.0) * z_max_m
    return H

def compute_slopes(H: np.ndarray, Lx_m: float, Ly_m: float):
    """
    Compute terrain gradients and slope metrics.
    H: height in meters, shape (Ny,Nx)
    Returns:
      dx, dy (grid spacing), gx, gy (gradients), grade, slope_rad, slope_deg
    """
    Ny, Nx = H.shape
    # Heightmap convention: N samples spanning full length L -> spacing L/(N-1)
    dx = Lx_m / (Nx - 1)
    dy = Ly_m / (Ny - 1)

    # Central differences via numpy gradient (2nd order in interior)
    dH_dy, dH_dx = np.gradient(H, dy, dx)  # note order: y then x

    # Grade magnitude (unitless): sqrt((dz/dx)^2 + (dz/dy)^2)
    grade = np.sqrt(dH_dx**2 + dH_dy**2)

    # Slope angle: atan(grade)
    slope_rad = np.arctan(grade)
    slope_deg = np.degrees(slope_rad)

    return dx, dy, dH_dx, dH_dy, grade, slope_rad, slope_deg

def summarize_region(H: np.ndarray, Lx_m: float, Ly_m: float, name: str = "region"):
    _, _, _, _, grade, _, slope_deg = compute_slopes(H, Lx_m, Ly_m)

    z_min = float(np.min(H))
    z_max = float(np.max(H))
    dz_range = z_max - z_min

    max_slope_deg = float(np.max(slope_deg))
    rms_slope_deg = float(np.sqrt(np.mean(slope_deg**2)))
    p95_slope_deg = float(np.percentile(slope_deg, 95))

    rms_grade = float(np.sqrt(np.mean(grade**2)))  # unitless
    mean_grade = float(np.mean(grade))             # unitless
    max_grade = float(np.max(grade))               # unitless

    out = {
        "name": name,
        "z_min_m": z_min,
        "z_max_m": z_max,
        "dz_range_m": dz_range,
        "max_slope_deg": max_slope_deg,
        "rms_slope_deg": rms_slope_deg,
        "p95_slope_deg": p95_slope_deg,
        "mean_grade_pct": 100.0 * mean_grade,
        "rms_grade_pct": 100.0 * rms_grade,
        "max_grade_pct": 100.0 * max_grade,
    }
    return out

def crop_roi_center(H_full: np.ndarray, L_full_m: float, roi_m: float):
    """
    Center crop a square ROI of physical size roi_m x roi_m from a square full map L_full_m x L_full_m.
    """
    N = H_full.shape[0]
    assert H_full.shape[0] == H_full.shape[1], "Expected square heightmap."
    dx = L_full_m / (N - 1)
    roi_px = int(round(roi_m / dx)) + 1  # include endpoints
    roi_px = min(roi_px, N)

    c = N // 2
    half = roi_px // 2
    r0 = max(0, c - half)
    r1 = min(N, r0 + roi_px)
    r0 = r1 - roi_px
    roi = H_full[r0:r1, r0:r1]
    return roi

def crop_roi_by_bounds(H_full: np.ndarray, L_full_m: float, x0_m: float, x1_m: float, y0_m: float, y1_m: float):
    """
    Crop ROI given physical bounds in meters in the heightmap coordinate frame:
      x in [0, L_full_m], y in [0, L_full_m]
    """
    N = H_full.shape[0]
    dx = L_full_m / (N - 1)

    def m_to_i(m):  # meters -> pixel index
        return int(round(m / dx))

    i0 = np.clip(m_to_i(x0_m), 0, N-1)
    i1 = np.clip(m_to_i(x1_m), 0, N-1)
    j0 = np.clip(m_to_i(y0_m), 0, N-1)
    j1 = np.clip(m_to_i(y1_m), 0, N-1)

    if i1 <= i0 or j1 <= j0:
        raise ValueError("Invalid ROI bounds after conversion to indices.")

    return H_full[j0:j1+1, i0:i1+1]

def pretty_print(stats: dict):
    print(f"\n=== Terrain metrics: {stats['name']} ===")
    print(f"z_min / z_max: {stats['z_min_m']:.3f} m / {stats['z_max_m']:.3f} m")
    print(f"Δz range:      {stats['dz_range_m']:.3f} m")
    print(f"max slope:     {stats['max_slope_deg']:.2f} deg")
    print(f"RMS slope:     {stats['rms_slope_deg']:.2f} deg")
    print(f"95% slope:     {stats['p95_slope_deg']:.2f} deg")
    print(f"mean grade:    {stats['mean_grade_pct']:.2f} %")
    print(f"RMS grade:     {stats['rms_grade_pct']:.2f} %")
    print(f"max grade:     {stats['max_grade_pct']:.2f} %")

def main():
    # --- user parameters ---
    png_path = "terrain1_1025.png"  # <-- change if needed
    L_full_m = 45.0
    z_max_m = 2.5

    # ROI options:
    roi_size_m = 8.0  # your experiment area
    use_center_roi = False

    # If not centered, set bounds (meters) in heightmap frame:
    # x0,x1,y0,y1 in [0, L_full_m]
    roi_bounds = (0.0, 8.0, 0.0, 8.0)  # example; only used if use_center_roi=False

    H_full = load_heightmap_png(png_path, z_max_m=z_max_m)

    # Full-map stats
    full_stats = summarize_region(H_full, L_full_m, L_full_m, name="full 45x45 m")
    pretty_print(full_stats)

    # ROI stats
    if use_center_roi:
        H_roi = crop_roi_center(H_full, L_full_m, roi_size_m)
        roi_stats = summarize_region(H_roi, roi_size_m, roi_size_m, name=f"ROI {roi_size_m}x{roi_size_m} m (center)")
    else:
        x0, x1, y0, y1 = roi_bounds
        H_roi = crop_roi_by_bounds(H_full, L_full_m, x0, x1, y0, y1)
        roi_stats = summarize_region(H_roi, x1-x0, y1-y0, name=f"ROI [{x0},{x1}]x[{y0},{y1}] m")

    pretty_print(roi_stats)

if __name__ == "__main__":
    main()