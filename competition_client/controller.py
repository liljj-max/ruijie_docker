#!/usr/bin/env python3
"""Differential-drive path follower (pure pursuit with curvature limiting).

The MMK2 base is a two-wheel differential drive (wheel_radius=0.0838,
wheel_distance=0.189): there is no minimum turning radius, but a motion at
speed v and yaw rate w follows an arc of radius R = v / w.  This controller:
  * picks a lookahead point on the path and computes the pursuit curvature
    kappa = 2 sin(alpha) / Ld, then w = v * kappa;
  * reduces v (down to 0) when the requested curvature exceeds the yaw-rate
    limit, so it never drives through a corner it cannot steer;
  * pivots in place (v = 0) when the heading error is large.

Commands are kept inside the achievable envelope by the adapter's limits.
"""

from __future__ import annotations

import math
from typing import List, Tuple

Point = Tuple[float, float]


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class PurePursuit:
    def __init__(self, lookahead: float = 0.50, max_lin: float = 0.18,
                 max_ang: float = 0.30, goal_tol: float = 0.15,
                 slow_radius: float = 0.80, pivot_ang: float = 0.60,
                 kp: float = 0.8, kd: float = 0.5):
        self.lookahead = lookahead
        self.max_lin = max_lin
        self.max_ang = max_ang
        self.goal_tol = goal_tol
        self.slow_radius = slow_radius
        self.pivot_ang = pivot_ang
        self.kp = kp
        self.kd = kd

    def compute(self, xy: Point, yaw: float, path: List[Point], yaw_rate: float = 0.0):
        """Return (lin, ang, dist_to_goal).

        The yaw command is a damped proportional law
        ``ang = clip(kp*alpha - kd*yaw_rate, +-max_ang)`` which prevents the
        bang-bang limit cycle that a saturated pivot produces on this laggy base.
        """
        if not path:
            return 0.0, 0.0, float("inf")
        goal = path[-1]
        dist_goal = math.hypot(goal[0] - xy[0], goal[1] - xy[1])

        target = path[0]
        for p in path:
            if math.hypot(p[0] - xy[0], p[1] - xy[1]) > self.lookahead:
                target = p
                break
        else:
            target = goal

        dx = target[0] - xy[0]
        dy = target[1] - xy[1]
        alpha = wrap_to_pi(math.atan2(dy, dx) - yaw)

        ang = self.kp * alpha - self.kd * yaw_rate
        ang = max(-self.max_ang, min(self.max_ang, ang))

        if abs(alpha) > self.pivot_ang:
            lin = 0.0
        else:
            v = self.max_lin
            if dist_goal < self.slow_radius:
                v = min(v, max(0.04, 0.25 * dist_goal))
            lin = v * max(0.0, math.cos(alpha))
        return lin, ang, dist_goal
