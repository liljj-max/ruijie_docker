#!/usr/bin/env python3
"""DWB-style local planner for the differential MMK2 base.

Critics follow the Nav2 DWB idea: sample (v, w), forward-simulate a short
differential-drive trajectory, discard collisions, then score by
  * PathProgress  : arc length advanced along the global path (dominant)
  * PathAlign     : heading aligned with the path tangent
  * PathDist      : distance to the path
  * GoalDist      : distance to the goal (fine approach)
  * Clearance     : margin to obstacles
  * Speed         : prefer forward speed
  * Smoothness    : penalise command changes and |w|

Only forward velocities are sampled while following; reverse is reserved for
the recovery/escape behaviour.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

Point = Tuple[float, float]


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class DWBLocalPlanner:
    def __init__(self, v_samples=None, w_samples=None,
                 horizon: float = 1.5, dt: float = 0.1, lookahead: float = 0.45,
                 w_progress: float = 4.0, w_align: float = 1.5,
                 w_pathdist: float = 1.5, w_goal: float = 1.0,
                 w_clear: float = 0.6, w_speed: float = 0.8,
                 w_omega: float = 0.2, w_smooth: float = 0.4,
                 w_osc: float = 0.3, w_prefer_fwd: float = 0.3,
                 w_rotate: float = 1.0, safety: float = 0.05):
        self.v_samples = list(v_samples) if v_samples is not None else \
            [0.0, 0.03, 0.06, 0.09, 0.12]
        self.w_samples = list(w_samples) if w_samples is not None else \
            [i * 0.25 / 3.0 for i in range(-3, 4)]
        self.horizon = horizon
        self.dt = dt
        self.lookahead = lookahead
        self.w_progress = w_progress
        self.w_align = w_align
        self.w_pathdist = w_pathdist
        self.w_goal = w_goal
        self.w_clear = w_clear
        self.w_speed = w_speed
        self.w_omega = w_omega
        self.w_smooth = w_smooth
        self.w_osc = w_osc
        self.w_prefer_fwd = w_prefer_fwd
        self.w_rotate = w_rotate
        self.safety = safety
        self.prev_v = 0.0
        self.prev_w = 0.0

    def reset(self) -> None:
        self.prev_v = 0.0
        self.prev_w = 0.0

    # ---- path helpers ----
    @staticmethod
    def _cum_len(path):
        cum = [0.0]
        for k in range(len(path) - 1):
            cum.append(cum[-1] + math.hypot(path[k + 1][0] - path[k][0],
                                            path[k + 1][1] - path[k][1]))
        return cum

    @staticmethod
    def _project(xy, path, cum):
        """Return (dist, arc_len, tangent_angle) of the closest point on the path."""
        best = (float("inf"), 0.0, 0.0)
        for k in range(len(path) - 1):
            ax, ay = path[k]
            bx, by = path[k + 1]
            vx, vy = bx - ax, by - ay
            l2 = vx * vx + vy * vy
            if l2 < 1e-9:
                continue
            t = ((xy[0] - ax) * vx + (xy[1] - ay) * vy) / l2
            t = max(0.0, min(1.0, t))
            px, py = ax + vx * t, ay + vy * t
            d = math.hypot(xy[0] - px, xy[1] - py)
            if d < best[0]:
                best = (d, cum[k] + t * math.sqrt(l2), math.atan2(vy, vx))
        return best

    def _simulate(self, planner, xy, yaw, v, w):
        x, y, th = xy[0], xy[1], yaw
        min_clear = float("inf")
        steps = max(1, int(self.horizon / self.dt))
        for _ in range(steps):
            x += v * math.cos(th) * self.dt
            y += v * math.sin(th) * self.dt
            th += w * self.dt
            i, j = planner.world_to_idx(x, y)
            if not (0 <= i < planner.nx and 0 <= j < planner.ny):
                return False, None, 0.0
            m = float(planner.margin[i, j])
            if m < self.safety:
                return False, None, 0.0
            if m < min_clear:
                min_clear = m
        return True, (x, y, th), min_clear

    def compute(self, planner, xy: Point, yaw: float, path: List[Point],
                goal: Point, allow_reverse: bool = False):
        """Return (v, w, dist_to_goal).  (0, 0) if no feasible sample."""
        if not path:
            return 0.0, 0.0, float("inf")
        dist_goal = math.hypot(goal[0] - xy[0], goal[1] - xy[1])
        cum = self._cum_len(path)

        v_samples = list(self.v_samples)
        if allow_reverse:
            v_samples += [-0.10, -0.15]

        # In tight spots (low clearance) allow deviating from the path to find a
        # wider route, and weight clearance more.
        ci, cj = planner.world_to_idx(xy[0], xy[1])
        cur_m = float(planner.margin[ci, cj]) if (0 <= ci < planner.nx and 0 <= cj < planner.ny) else 0.0
        tight = cur_m < 0.15
        w_pathdist = self.w_pathdist * (0.3 if tight else 1.0)
        w_clear = self.w_clear * (2.5 if tight else 1.0)

        best: Optional[Tuple[float, float, float]] = None
        for v in v_samples:
            for w in self.w_samples:
                ok, end, clear = self._simulate(planner, xy, yaw, v, w)
                if not ok:
                    continue
                pd, s_end, tangent = self._project(end, path, cum)
                align = abs(wrap_to_pi(tangent - end[2]))
                gd = math.hypot(goal[0] - end[0], goal[1] - end[1])
                score = (self.w_progress * s_end
                         - self.w_align * align
                         - w_pathdist * pd
                         - self.w_goal * gd
                         + w_clear * min(clear, 0.5)
                         + self.w_speed * (v if v > 0 else 0.0)
                         - self.w_omega * abs(w)
                         - self.w_smooth * (abs(v - self.prev_v) + abs(w - self.prev_w)))
                # PreferForward: discourage standing still while following
                if not allow_reverse and v <= 0.0:
                    score -= self.w_prefer_fwd
                # Oscillation: discourage flipping the turn direction
                if self.prev_w * w < 0.0 and abs(self.prev_w) > 0.05:
                    score -= self.w_osc
                # RotateToGoal: near the goal, prefer slowing translation
                if dist_goal < 0.30:
                    score -= self.w_rotate * abs(v)
                if best is None or score > best[0]:
                    best = (score, v, w)

        if best is None:
            self.prev_v = self.prev_w = 0.0
            return 0.0, 0.0, dist_goal
        self.prev_v, self.prev_w = best[1], best[2]
        return best[1], best[2], dist_goal
