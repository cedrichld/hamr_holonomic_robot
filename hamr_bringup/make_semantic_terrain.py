#!/usr/bin/env python3
import numpy as np
from PIL import Image, ImageFilter
from pathlib import Path
import json, random

# --- CONFIG ----------------------------------------------------
# 1) classes you want to start with (add more later)
#    Put 1–3 tileable images in each folder; script will pick one at random for variety.
CLASS_SPEC = {
    #  "name"       : {"rgb": (R,G,B),       "tex_dir": "textures/name", "meters_per_tile": 1.0}
    "flagstone"     : {"rgb": (237,124, 59), "tex_dir": "terrain_assets/textures/flagstone", "meters_per_tile": 0.6},
    "pebble"        : {"rgb": (255,190,122), "tex_dir": "terrain_assets/textures/pebble",    "meters_per_tile": 0.35},
    "boulder"       : {"rgb": (207, 89,247), "tex_dir": "terrain_assets/textures/boulder",   "meters_per_tile": 0.8},
    "grass"         : {"rgb": (  0, 84,  0), "tex_dir": "terrain_assets/textures/grass",     "meters_per_tile": 0.5},
    "sand"          : {"rgb": (235,182, 74), "tex_dir": "terrain_assets/textures/sand",      "meters_per_tile": 0.7},
    "stone_dust"    : {"rgb": (240,236,196), "tex_dir": "terrain_assets/textures/stone_dust","meters_per_tile": 0.5},
    "wood"          : {"rgb": (108,  0, 26), "tex_dir": "terrain_assets/textures/wood",      "meters_per_tile": 0.7},
    "foam"          : {"rgb": ( 96,150,255), "tex_dir": "terrain_assets/textures/foam",      "meters_per_tile": 0.6},
    "concrete"      : {"rgb": (110,110,110), "tex_dir": "terrain_assets/textures/concrete",  "meters_per_tile": 0.8},
    "tree"          : {"rgb": (  0,  0,  0), "tex_dir": "terrain_assets/textures/tree",      "meters_per_tile": 0.6},
}
# 2) world size (meters) covered by the image
WORLD_SIZE_M = (8.0, 8.0)  # (x,y). Change to your arena size.
# 3) feather radius (pixels)
FEATHER_PX = 6
# ---------------------------------------------------------------

def load_rgba(path):
    img = Image.open(path).convert('RGBA')
    return img

def pick_texture(dirpath: Path, kind='albedo'):
    # prefer files containing 'albedo' or 'color'; else any png/jpg
    cands = sorted([p for p in dirpath.glob('*.png')] + [p for p in dirpath.glob('*.jpg')])
    pref  = [p for p in cands if 'albedo' in p.stem.lower() or 'color' in p.stem.lower()]
    pool  = pref if pref else cands
    if not pool:
        raise FileNotFoundError(f"No textures found in {dirpath}")
    return random.choice(pool)

def tile_to_canvas(tex_img: Image.Image, out_w, out_h, meters_per_tile, world_size_m):
    # compute tile size in pixels from meters_per_tile
    px_per_m_x = out_w / world_size_m[0]
    px_per_m_y = out_h / world_size_m[1]
    tile_w = max(8, int(round(meters_per_tile * px_per_m_x)))
    tile_h = max(8, int(round(meters_per_tile * px_per_m_y)))
    tile = tex_img.resize((tile_w, tile_h), Image.BICUBIC)
    cols = (out_w + tile_w - 1) // tile_w
    rows = (out_h + tile_h - 1) // tile_h
    big = Image.new('RGBA', (cols*tile_w, rows*tile_h))
    for r in range(rows):
        for c in range(cols):
            big.paste(tile, (c*tile_w, r*tile_h))
    return big.crop((0,0,out_w,out_h))

def rgb_nearest(labels_rgb, palette):
    # labels_rgb: (H,W,3) uint8; palette: list of (name, (r,g,b))
    lab = labels_rgb.astype(np.int16)
    pal = np.array([p[1] for p in palette], dtype=np.int16)  # Kx3
    # compute squared distance to each palette color
    diff = lab.reshape(-1,1,3) - pal.reshape(1,-1,3)
    d2   = (diff*diff).sum(axis=2)  # (N,K)
    idx  = d2.argmin(axis=1).reshape(labels_rgb.shape[:2])
    return idx  # index per pixel

def gaussian_feather(mask_img, radius_px):
    if radius_px <= 0: return mask_img
    return mask_img.filter(ImageFilter.GaussianBlur(radius_px))

def composite_textures(label_img_path, out_albedo='albedo_out.png', out_normal=None):
    labels = Image.open(label_img_path).convert('RGB')
    H, W = labels.size[1], labels.size[0]
    palette = [(name, spec["rgb"]) for name, spec in CLASS_SPEC.items()]
    idx = rgb_nearest(np.array(labels), palette)  # (H,W) ints

    # optional: snap labels to reduce speckles
    # simple 3x3 majority filter
    lab = idx.copy()
    from collections import Counter
    padded = np.pad(lab, 1, mode='edge')
    for y in range(H):
        for x in range(W):
            window = padded[y:y+3, x:x+3].ravel()
            lab[y,x] = Counter(window).most_common(1)[0][0]
    idx = lab

    # base canvases
    comp_alb = Image.new('RGBA', (W,H), (0,0,0,255))
    comp_nrm = Image.new('RGBA', (W,H), (128,128,255,255)) if out_normal else None

    for k, (name, _) in enumerate(palette):
        mask = Image.fromarray((idx==k).astype(np.uint8)*255, mode='L')
        if mask.getbbox() is None:
            continue  # class not present
        spec = CLASS_SPEC[name]
        tex_dir = Path(spec["tex_dir"])
        alb_path = pick_texture(tex_dir, 'albedo')
        alb = load_rgba(alb_path)
        tiled_alb = tile_to_canvas(alb, W, H, spec["meters_per_tile"], WORLD_SIZE_M)

        # feather edges
        mask_soft = gaussian_feather(mask, FEATHER_PX)

        # composite
        comp_alb = Image.composite(tiled_alb, comp_alb, mask_soft)

        if comp_nrm:
            # if a normal exists, tile it; else synthesize a flat normal for that class
            nrm_files = list(tex_dir.glob('*normal*.png')) + list(tex_dir.glob('*nrm*.png'))
            if nrm_files:
                nrm = load_rgba(nrm_files[0])
            else:
                nrm = Image.new('RGBA', (64,64), (128,128,255,255))  # flat normal
            tiled_nrm = tile_to_canvas(nrm, W, H, spec["meters_per_tile"], WORLD_SIZE_M)
            comp_nrm = Image.composite(tiled_nrm, comp_nrm, mask_soft)

    comp_alb.save(out_albedo)
    if comp_nrm and out_normal:
        comp_nrm.save(out_normal)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels",  required=True, help="semantic color map (PNG)")
    # ap.add_argument("--height",  required=True, help="grayscale heightmap (PNG)")
    ap.add_argument("--albedo_out", default="albedo_out.png")
    ap.add_argument("--normal_out", default=None)  # set to filename to export normals
    args = ap.parse_args()
    composite_textures(args.labels, args.albedo_out, args.normal_out)
    print("Wrote", args.albedo_out, "and", args.normal_out)
