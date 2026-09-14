#!/usr/bin/env python3
"""YOLO detector backend for the 9 supermarket product kinds.

Loads a standard YOLOv8 checkpoint (trained via the official toolchain) and
returns detections with the product kind name.  Compatible with the client
image's ultralytics 8.0.196 + PyTorch >= 2.6 (weights_only patch).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict

import numpy as np

DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "products9.pt"

# Fallback mapping if the checkpoint does not embed names.
KIND_NAMES = [
    "chengzi", "heweidao", "kele", "kouxiangtang", "maidong",
    "pingguo", "sanmingzhi", "shupian", "zhijin",
]


class ProductDetector:
    def __init__(self, weights=DEFAULT_WEIGHTS, confidence: float = 0.35,
                 device: str = "auto"):
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        self.confidence = float(confidence)

        import torch
        from ultralytics import YOLO

        selected = self._select_device(torch, device)
        orig = torch.load

        def compat(*a, **kw):
            kw.setdefault("weights_only", False)
            return orig(*a, **kw)

        torch.load = compat
        try:
            self.model = YOLO(str(weights)).to(selected)
            self.model.model.eval()
        finally:
            torch.load = orig

        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            self.names = [names[i] for i in sorted(names)]
        elif isinstance(names, (list, tuple)) and names:
            self.names = list(names)
        else:
            self.names = list(KIND_NAMES)
        print(f"[ProductDetector] {weights.name} on {selected}, classes={self.names}")

    @staticmethod
    def _select_device(torch, requested: str):
        requested = requested.lower()
        if requested == "cpu" or not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device("cuda:0")

    def detect(self, rgb: np.ndarray) -> List[Dict]:
        results = self.model(rgb, verbose=False)[0]
        dets: List[Dict] = []
        for box in results.boxes:
            conf = float(box.conf.item())
            if conf < self.confidence:
                continue
            cls = int(box.cls.item())
            if cls >= len(self.names):
                continue
            x0, y0, x1, y1 = map(int, box.xyxy[0].cpu().numpy())
            dets.append({
                "kind": self.names[cls],
                "class_id": cls,
                "x": (x0 + x1) // 2,
                "y": (y0 + y1) // 2,
                "w": x1 - x0,
                "h": y1 - y0,
                "conf": conf,
            })
        return dets
