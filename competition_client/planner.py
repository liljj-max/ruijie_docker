#!/usr/bin/env python3
"""Lightweight occupancy-grid planner (A*) for the supermarket scene.

Replaces the (unavailable) Nav2 SmacPlanner2D with a self-contained Python
implementation:
  * a static grid built from the PUBLIC fixed scene structure, split by height:
      - HIGH (shelves, perimeter walls, corridor board, ~1.3-1.5 m tall) is
        inflated by the ARM reach (the MMK2 arms stick out ~0.44-0.47 m at
        z~1.2 m, measured from the collision meshes);
      - LOW (delivery table ~0.77 m, random boxes ~0.5 m) is inflated by the
        chassis radius only, because the stowed arms pass above them;
  * a dynamic layer from the live 360 LaserScan (ray casting with marking and
    clearing) plus a footprint-box filter for self returns;
  * cost-weighted 8-connected A* with line-of-sight smoothing.

No server truth (product/slot layout) is read.
"""

from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage

Point = Tuple[float, float]

# Public fixed scene geometry (world frame, from retail_competition.xml).
FIELD_X = (-2.5, 2.5)
FIELD_Y = (-3.75, 3.75)
SHELF_XS = (-1.735, -0.850, 0.035, 0.920, 1.805)
SHELF_Y = 3.323
DELIVERY_TABLE = (-2.42, -1.46, -3.63, -3.19)   # x0, x1, y0, y1  (LOW, 0.77 m)
CORRIDOR_BOARD = (0.515, 0.545, -3.72, 1.70)    # x0, x1, y0, y1  (HIGH, 1.5 m)

# Inflation radii (m).  Mapped from the previous Nav2 setup: the global planner
# used robot_radius 0.35 / footprint 0.38 with a soft inflation_radius 0.60-0.65
# (paths allowed to pass closer, cost gradient pushes them away).  HIGH covers
# the arm reach, LOW only the chassis (the stowed arms pass above low obstacles).
INFLATE_HIGH = 0.45
# LOW obstacles (delivery table, random boxes) are inflated by the chassis
# radius only: 0.35 closes the ~0.59 m fixed-layout corridor gap, 0.25 keeps it
# passable while the DWB still holds a 0.05 m safety margin.
INFLATE_LOW = 0.25

# Used only by callers that cannot provide the live base_link->laser TF.
LASER_OFFSET = (0.1137, 0.0, 0.0)
# Footprint half extents used to drop self returns (range_filter footprint mode).
FOOT_HALF_X = 0.35
FOOT_HALF_Y = 0.35
# A return is only treated as a self echo when it is inside the footprint box
# AND at/behind the lidar face (plus this small pad).  The lidar sits 0.1137 m
# ahead of base_link, so a real obstacle that reaches the front of the box
# (e.g. box_05 at ~0.23 m) must NOT be discarded.
SELF_ECHO_PAD = 0.04

MIN_RANGE = 0.05
MAX_RANGE = 12.0
PERSIST = 8            # frames an obstacle survives without re-observation
PREFER_CLEAR = 0.60    # prefer this much margin beyond the inflation radii
COST_K = 8.0           # weight of the proximity cost (higher = keep to the middle)
# A* must not plan through gaps the local planner refuses to follow: keep the
# same clearance the DWB uses as its collision safety.  It also has to stay
# within the server's guaranteed traversability (obstacle_layout's
# ROBOT_CLEARANCE_RADIUS = 0.347 m): a larger margin (0.05) made the global
# planner reject random layouts the server considers passable, returning no
# path.  0.02 keeps INFLATE_HIGH + BLOCK_MARGIN = 0.47, still covering the
# stowed arm reach (~0.435 m) near the corridor board / walls.
BLOCK_MARGIN = 0.02


class GridPlanner:
    def __init__(self, res: float = 0.05, margin: float = 0.15):
        self.res = float(res)
        self.xmin = FIELD_X[0] - margin
        self.ymin = FIELD_Y[0] - margin
        self.nx = int(round((FIELD_X[1] - FIELD_X[0] + 2 * margin) / self.res))
        self.ny = int(round((FIELD_Y[1] - FIELD_Y[0] + 2 * margin) / self.res))
        self.static_high = np.zeros((self.nx, self.ny), dtype=bool)
        self.static_low = np.zeros((self.nx, self.ny), dtype=bool)
        self.dynamic = np.zeros((self.nx, self.ny), dtype=bool)
        self.hits = np.zeros((self.nx, self.ny), dtype=np.int16)
        self.blocked = np.zeros((self.nx, self.ny), dtype=bool)
        self.margin = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.dist_all = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.cost = np.ones((self.nx, self.ny), dtype=np.float32)
        self._build_static()
        self._recompute()

    # ---- indexing ----
    def world_to_idx(self, x: float, y: float) -> Tuple[int, int]:
        return int((x - self.xmin) / self.res), int((y - self.ymin) / self.res)

    def idx_to_world(self, i: int, j: int) -> Point:
        return (self.xmin + (i + 0.5) * self.res,
                self.ymin + (j + 0.5) * self.res)

    def _mark_rect(self, layer, x0, x1, y0, y1) -> None:
        i0, j0 = self.world_to_idx(x0, y0)
        i1, j1 = self.world_to_idx(x1, y1)
        i0, j0 = max(0, i0), max(0, j0)
        i1, j1 = min(self.nx - 1, i1), min(self.ny - 1, j1)
        if i1 >= i0 and j1 >= j0:
            layer[i0:i1 + 1, j0:j1 + 1] = True

    def _build_static(self) -> None:
        h, lo = self.static_high, self.static_low
        # perimeter walls (HIGH)
        self._mark_rect(h, FIELD_X[0] - 0.15, FIELD_X[1] + 0.15, 3.70, FIELD_Y[1] + 0.15)
        self._mark_rect(h, FIELD_X[0] - 0.15, FIELD_X[1] + 0.15, FIELD_Y[0] - 0.15, -3.70)
        self._mark_rect(h, FIELD_X[0] - 0.15, -2.46, FIELD_Y[0] - 0.15, FIELD_Y[1] + 0.15)
        self._mark_rect(h, 2.46, FIELD_X[1] + 0.15, FIELD_Y[0] - 0.15, FIELD_Y[1] + 0.15)
        # shelves (HIGH)
        for sx in SHELF_XS:
            self._mark_rect(h, sx - 0.46, sx + 0.46, SHELF_Y - 0.19, SHELF_Y + 0.19)
        # corridor board (HIGH)
        self._mark_rect(h, *CORRIDOR_BOARD)
        # delivery table (LOW)
        self._mark_rect(lo, *DELIVERY_TABLE)

    def _recompute(self) -> None:
        d_high = ndimage.distance_transform_edt(~self.static_high) * self.res
        d_low = ndimage.distance_transform_edt(~(self.static_low | self.dynamic)) * self.res
        self.margin = np.minimum(d_high - INFLATE_HIGH,
                                 d_low - INFLATE_LOW).astype(np.float32)
        self.blocked = self.margin <= BLOCK_MARGIN
        occ_all = self.static_high | self.static_low | self.dynamic
        self.dist_all = (ndimage.distance_transform_edt(~occ_all) * self.res).astype(np.float32)
        self.cost = 1.0 + COST_K * np.clip(
            1.0 - np.maximum(self.margin, 0.0) / PREFER_CLEAR, 0.0, 1.0)

    def grad_at(self, i: int, j: int):
        """Grid gradient of the obstacle-distance field (points away from obstacles)."""
        ip, im = min(i + 1, self.nx - 1), max(i - 1, 0)
        jp, jm = min(j + 1, self.ny - 1), max(j - 1, 0)
        di = float(self.dist_all[ip, j] - self.dist_all[im, j])
        dj = float(self.dist_all[i, jp] - self.dist_all[i, jm])
        return di, dj

    # ---- dynamic layer from the 360 LaserScan ----
    def update_scan(self, ranges, angle_min: float, angle_inc: float,
                    rx: float, ry: float, ryaw: float,
                    sensor_pose=LASER_OFFSET, range_min: float = MIN_RANGE,
                    range_max: float = MAX_RANGE) -> None:
        """Ray-cast the scan into the dynamic layer (marking + clearing)."""
        self.hits = np.maximum(self.hits - 1, 0)
        if ranges is None or len(ranges) == 0:
            self.dynamic = self.hits > 0
            self._recompute()
            return

        r = np.asarray(ranges, dtype=np.float64)
        n = r.size
        ang = angle_min + angle_inc * np.arange(n)

        sx, sy, syaw = sensor_pose
        lx = rx + math.cos(ryaw) * sx - math.sin(ryaw) * sy
        ly = ry + math.sin(ryaw) * sx + math.cos(ryaw) * sy

        min_r = max(MIN_RANGE, float(range_min))
        max_r = min(MAX_RANGE, float(range_max))
        valid = np.isfinite(r) & (r > min_r) & (r < max_r)
        bang = syaw + ang
        ex_b = sx + r * np.cos(bang)
        ey_b = sy + r * np.sin(bang)
        inside = (np.abs(ex_b) <= FOOT_HALF_X) & (np.abs(ey_b) <= FOOT_HALF_Y)
        behind_lidar_face = ex_b <= sx + SELF_ECHO_PAD
        valid &= ~(inside & behind_lidar_face)
        if not np.any(valid):
            self.dynamic = self.hits > 0
            self._recompute()
            return

        wang = ryaw + bang
        rv = np.where(valid, r, 0.0)

        d = np.arange(0.0, MAX_RANGE, self.res)
        D = d[None, :]
        free = (D < rv[:, None]) & valid[:, None]
        hit = (D <= rv[:, None]) & (rv[:, None] < D + self.res) & valid[:, None]

        X = lx + D * np.cos(wang)[:, None]
        Y = ly + D * np.sin(wang)[:, None]
        I = ((X - self.xmin) / self.res).astype(np.int32)
        J = ((Y - self.ymin) / self.res).astype(np.int32)
        ingrid = (I >= 0) & (I < self.nx) & (J >= 0) & (J < self.ny)

        fm = free & ingrid
        hm = hit & ingrid
        self.hits[I[fm], J[fm]] = 0
        self.hits[I[hm], J[hm]] = PERSIST
        self.dynamic = self.hits > 0
        self._recompute()

    # ---- planning ----
    def _nearest_free(self, i: int, j: int, max_r: int = 12) -> Optional[Tuple[int, int]]:
        if 0 <= i < self.nx and 0 <= j < self.ny and not self.blocked[i, j]:
            return i, j
        for rr in range(1, max_r + 1):
            for di in range(-rr, rr + 1):
                for dj in range(-rr, rr + 1):
                    if max(abs(di), abs(dj)) != rr:
                        continue
                    ni, nj = i + di, j + dj
                    if (0 <= ni < self.nx and 0 <= nj < self.ny
                            and not self.blocked[ni, nj]):
                        return ni, nj
        return None

    def plan(self, start: Point, goal: Point) -> Optional[List[Point]]:
        si, sj = self.world_to_idx(*start)
        gi, gj = self.world_to_idx(*goal)
        s = self._nearest_free(si, sj)
        g = self._nearest_free(gi, gj)
        if s is None or g is None:
            return None
        si, sj = s
        gi, gj = g

        dirs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                (-1, -1, 1.4142), (-1, 1, 1.4142),
                (1, -1, 1.4142), (1, 1, 1.4142)]
        open_set = [(0.0, si, sj)]
        came = {}
        gcost = {(si, sj): 0.0}
        found = (si, sj) == (gi, gj)
        while open_set:
            _, i, j = heapq.heappop(open_set)
            if (i, j) == (gi, gj):
                found = True
                break
            for di, dj, c in dirs:
                ni, nj = i + di, j + dj
                if not (0 <= ni < self.nx and 0 <= nj < self.ny):
                    continue
                if self.blocked[ni, nj]:
                    continue
                if di != 0 and dj != 0:
                    if self.blocked[i + di, j] or self.blocked[i, j + dj]:
                        continue  # no corner cutting
                ng = gcost[(i, j)] + c * float(self.cost[ni, nj])
                if ng < gcost.get((ni, nj), 1e18):
                    gcost[(ni, nj)] = ng
                    came[(ni, nj)] = (i, j)
                    h = math.hypot(ni - gi, nj - gj)
                    heapq.heappush(open_set, (ng + h, ni, nj))
        if not found:
            return None
        path_idx = [(gi, gj)]
        cur = (gi, gj)
        while cur != (si, sj):
            cur = came[cur]
            path_idx.append(cur)
        path_idx.reverse()
        pts = [self.idx_to_world(i, j) for i, j in path_idx]
        pts = self._smooth(pts)
        return self._clear_path(pts)

    def _clear_path(self, pts, target: float = 0.20, iters: int = 6):
        """Push interior path points away from obstacles to raise clearance.

        This stops the path from hugging the inflated boundary at corners (which
        otherwise makes the robot graze the boundary and oscillate)."""
        if len(pts) <= 2:
            return pts
        out = [np.array(p, dtype=float) for p in pts]
        for _ in range(iters):
            moved = False
            for k in range(1, len(out) - 1):
                x, y = out[k]
                i, j = self.world_to_idx(x, y)
                if not (0 <= i < self.nx and 0 <= j < self.ny):
                    continue
                m = float(self.margin[i, j])
                if m >= target:
                    continue
                di, dj = self.grad_at(i, j)
                n = math.hypot(di, dj)
                if n < 1e-6:
                    continue
                step = min(target - m, 0.05)
                nx = x + di / n * step
                ny = y + dj / n * step
                ii, jj = self.world_to_idx(nx, ny)
                if not (0 <= ii < self.nx and 0 <= jj < self.ny):
                    continue
                if self.margin[ii, jj] <= m:
                    continue
                if not (self._line_free((out[k - 1][0], out[k - 1][1]), (nx, ny))
                        and self._line_free((nx, ny), (out[k + 1][0], out[k + 1][1]))):
                    continue
                out[k] = np.array([nx, ny])
                moved = True
            if not moved:
                break
        return [(float(p[0]), float(p[1])) for p in out]

    def _line_free(self, a: Point, b: Point) -> bool:
        n = int(math.hypot(b[0] - a[0], b[1] - a[1]) / self.res) + 1
        for t in np.linspace(0.0, 1.0, n):
            x = a[0] + (b[0] - a[0]) * t
            y = a[1] + (b[1] - a[1]) * t
            i, j = self.world_to_idx(x, y)
            if not (0 <= i < self.nx and 0 <= j < self.ny) or self.blocked[i, j]:
                return False
        return True

    def _smooth(self, pts: List[Point]) -> List[Point]:
        if len(pts) <= 2:
            return pts
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self._line_free(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        return out

    def min_clearance(self, pts: List[Point]) -> float:
        """Minimum margin (>=0 means outside the inflated obstacles) on a path."""
        if not pts:
            return 0.0
        vals = []
        for x, y in pts:
            i, j = self.world_to_idx(x, y)
            if 0 <= i < self.nx and 0 <= j < self.ny:
                vals.append(float(self.margin[i, j]))
        return min(vals) if vals else 0.0
