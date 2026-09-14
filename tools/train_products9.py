#!/usr/bin/env python3
"""Fine-tune YOLOv8s on the 9-class product dataset.

Run INSIDE the client image (ultralytics 8.0.196 + GPU), e.g.:

    python3 /tools/train_products9.py \
        --data /out/dataset/data.yaml \
        --out /workspace/baseline/weights/products9.pt \
        --epochs 100 --imgsz 640 --batch 8
"""

import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/out/dataset/data.yaml")
    ap.add_argument("--weights", default="yolov8s.pt")
    ap.add_argument("--out", default="/workspace/baseline/weights/products9.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="/out/runs")
    ap.add_argument("--name", default="products9")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    orig = torch.load

    def compat(*a, **kw):
        kw.setdefault("weights_only", False)
        return orig(*a, **kw)

    torch.load = compat
    try:
        model = YOLO(args.weights)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            patience=args.patience,
            device=args.device,
            amp=False,
            project=args.project,
            name=args.name,
            exist_ok=True,
            verbose=True,
        )
        metrics = model.val()
    finally:
        torch.load = orig

    best = Path(args.project) / args.name / "weights" / "best.pt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if best.is_file():
        shutil.copy2(best, out)
        print(f"[train] copied {best} -> {out}")
    else:
        print(f"[train] WARNING: best.pt not found at {best}")

    try:
        print(f"[train] mAP50-95={metrics.box.map:.4f} mAP50={metrics.box.map50:.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"[train] metrics unavailable: {exc}")


if __name__ == "__main__":
    main()
