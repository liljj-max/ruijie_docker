#!/usr/bin/env python3
"""9-class YOLO dataset generator for the supermarket sorting task.

Adapted from the official ``perception/gen_dataset.py`` (single-class kele).
It renders the retail scene with the 3DGS renderer, poses the MMK2 head camera
near the shelf aisles, projects every product's ground-truth world position into
the image and auto-labels the visible ones.  All 45 products / 9 kinds are
labelled; the class index is the position of the kind in ``KIND_NAMES``.

Run INSIDE the server image (needs gsplat + discoverse + GPU), headless:

    cd /workspace/supermarket_sorting_task/examples/supermarket_sorting
    MUJOCO_GL=egl python3 /tools/gen_dataset_9cls.py --out /out --frames 400
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import cv2

TASK_DIR = Path(os.environ.get(
    "SUPERMARKET_TASK_DIR",
    "/workspace/supermarket_sorting_task/examples/supermarket_sorting"))
REPO_ROOT = TASK_DIR.parents[1]
for _p in (str(REPO_ROOT), str(TASK_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from perception import gen_dataset as gd  # noqa: E402

IMG_W, IMG_H = gd.IMG_W, gd.IMG_H
DEPTH_TOL = gd.DEPTH_TOL
MIN_WH, MAX_WH = gd.MIN_WH, gd.MAX_WH

KIND_NAMES = [
    "chengzi", "heweidao", "kele", "kouxiangtang", "maidong",
    "pingguo", "sanmingzhi", "shupian", "zhijin",
]
KIND_INDEX = {k: i for i, k in enumerate(KIND_NAMES)}

# Half extents (m) per kind, derived from the MJCF collision geometry.
HALF_EXTENTS = {
    "sanmingzhi": (0.033, 0.050, 0.050),
    "heweidao": (0.048, 0.048, 0.053),
    "shupian": (0.033, 0.033, 0.105),
    "zhijin": (0.086, 0.043, 0.044),
    "maidong": (0.033, 0.033, 0.105),
    "kouxiangtang": (0.025, 0.025, 0.040),
    "pingguo": (0.035, 0.035, 0.035),
    "chengzi": (0.037, 0.037, 0.037),
    "kele": (0.027, 0.027, 0.073),
}


def projected_bbox(pw, K, T_cw, half, margin=1.10):
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners.append(np.asarray(pw) + np.array([sx, sy, sz]) * np.asarray(half))
    pts = [gd.project_world_to_px(c, K, T_cw) for c in corners]
    pts = [p for p in pts if p is not None]
    if len(pts) < 4:
        return None
    us = np.array([p[0] for p in pts])
    vs = np.array([p[1] for p in pts])
    x0, x1 = float(us.min()), float(us.max())
    y0, y1 = float(vs.min()), float(vs.max())
    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    bw, bh = (x1 - x0) * margin, (y1 - y0) * margin
    x0 = int(max(0, round(cx - bw * 0.5)))
    x1 = int(min(IMG_W - 1, round(cx + bw * 0.5)))
    y0 = int(max(0, round(cy - bh * 0.5)))
    y1 = int(min(IMG_H - 1, round(cy + bh * 0.5)))
    if x1 <= x0 or y1 <= y0:
        return None
    bw, bh = x1 - x0, y1 - y0
    if not (MIN_WH <= bw <= MAX_WH and MIN_WH <= bh <= MAX_WH):
        return None
    return x0, y0, x1, y1


def label_all(depth_m, K, T_cw, slots):
    """Return list of (class_index, x0, y0, x1, y1) for visible products."""
    out = []
    for slot in slots:
        kind = slot.get("object_kind")
        if kind not in KIND_INDEX:
            continue
        pw = slot["world_position"]
        proj = gd.project_world_to_px(pw, K, T_cw)
        if proj is None:
            continue
        u, v, zp = proj
        ui, vi = int(round(u)), int(round(v))
        if not (0 <= ui < IMG_W and 0 <= vi < IMG_H):
            continue
        d = gd._patch_depth(depth_m, ui, vi)
        if d <= 0.0 or abs(d - zp) > DEPTH_TOL:
            continue
        box = projected_bbox(pw, K, T_cw, HALF_EXTENTS[kind])
        if box is not None:
            out.append((KIND_INDEX[kind], *box))
    return out


def boxes_to_yolo(boxes):
    lines = []
    for cls, x0, y0, x1, y1 in boxes:
        cx = (x0 + x1) / 2.0 / IMG_W
        cy = (y0 + y1) / 2.0 / IMG_H
        bw = (x1 - x0) / IMG_W
        bh = (y1 - y0) / IMG_H
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines)


def save_sample(out_dir, split, name, img_bgr, boxes):
    cv2.imwrite(str(out_dir / "images" / split / f"{name}.jpg"), img_bgr)
    (out_dir / "labels" / split / f"{name}.txt").write_text(boxes_to_yolo(boxes))


def main():
    ap = argparse.ArgumentParser(description="generate 9-class YOLO dataset")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--pose-mode", default="wide", choices=["baseline", "wide"])
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--out", default="/out/dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    sim = gd.build_sim()
    K = gd.head_cam_K()
    slots = json.loads(gd.LAYOUT_JSON.read_text())
    print(f"[gen9] {len(slots)} slots, {len(KIND_NAMES)} classes; "
          f"{args.frames} frames x (1+{args.variants}) variants, pose={args.pose_mode}")

    n_frames = n_boxes = n_imgs = 0
    attempts = 0
    per_class = {k: 0 for k in KIND_NAMES}
    while n_frames < args.frames and attempts < args.frames * 4:
        attempts += 1
        base_xy, yaw, slide, pitch = gd.sample_pose(rng, args.pose_mode)
        gd.set_robot_pose(sim, base_xy, yaw, slide, pitch)
        sim.render()
        rgb = sim.img_rgb_obs_s[gd.HEAD_CAM_ID]
        depth_m = sim.img_depth_obs_s[gd.HEAD_CAM_ID]
        T_cw = gd.T_cam_world(sim)
        boxes = label_all(depth_m, K, T_cw, slots)
        if not boxes:
            continue

        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        split = "val" if rng.random() < args.val_frac else "train"
        base_name = f"f{n_frames:04d}"
        variants = [rgb_bgr] + gd.domain_randomise(rng, rgb_bgr, args.variants)
        for vi, img in enumerate(variants):
            save_sample(out_dir, split, f"{base_name}_v{vi}", img, boxes)
            n_imgs += 1
        for cls, *_ in boxes:
            per_class[KIND_NAMES[cls]] += 1
        n_frames += 1
        n_boxes += len(boxes)
        if n_frames % 25 == 0:
            print(f"[gen9] {n_frames}/{args.frames} frames, {n_imgs} imgs, {n_boxes} boxes")

    (out_dir / "data.yaml").write_text(
        f"path: {out_dir}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(KIND_NAMES)}\n"
        f"names: {KIND_NAMES}\n"
    )
    print(f"[gen9] done: {n_frames} frames, {n_imgs} images, {n_boxes} boxes")
    print(f"[gen9] per-class boxes: {per_class}")
    print(f"[gen9] data.yaml -> {out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
