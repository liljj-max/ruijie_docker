#!/usr/bin/env python3
"""Competition client: task -> scan -> navigate -> pick -> deliver -> loop.

Single-process node graph:
    CompetitionClient (this node, 30 Hz tick)
      + MMK2Adapter   (control + odom/joint feedback)
      + TaskListener  (latched /supermarket_sorting/task)

Perception runs as a separate process (competition_client.perception) and
publishes JSON on /competition/product_detections and /competition/aruco_detections.

Safety: any navigation failure or missing sensor -> zero velocity; arm is kept
stowed while the base moves; Ctrl+C publishes zero velocity.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import deque
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

from competition_client.controller import PurePursuit
from competition_client.local_planner import DWBLocalPlanner
from competition_client.mmk2_adapter import MMK2Adapter, GRIP_CLOSE, GRIP_OPEN, wrap_to_pi
from competition_client.planner import GridPlanner
from competition_client.shelf_scanner import ShelfInventory
from competition_client.task_parser import TaskListener

# ---- scene / route constants (world frame, +X east / +Y north) ----
YELLOW_MID_Y = 2.475
GRASP_YAW = math.pi / 2.0 - math.radians(11.0)
DELIVERY_YAW = -math.pi / 2.0
APPROACH_DX = 0.068          # base sits this far east of the target column
SHELF_X = {"A": -1.735, "B": -0.850, "C": 0.035, "D": 0.920, "E": 1.805}
COLUMN_DX = {"C1": -0.220, "C2": 0.0, "C3": 0.220}
SCAN_Y = 2.45                # observation lane in front of the shelves
SCAN_SLIDE = 0.11            # raise the head to shelf level while scanning
SCAN_PITCHES = [-0.35, -0.65, -0.95]
TABLE_APPROACH = [-1.88, -2.80]

# manipulation params (from the reference baseline)
HEAD_PITCH = -0.6
SLIDE_GRASP = 0.11
LIFT_AMOUNT = 0.05
DEPLOY_OFFSET = np.array([-0.011, -0.220, -0.010])
CREEP_STOP_DY = 0.035
CREEP_SPEED = 0.10
RETREAT_SPEED = 0.20
PLACE_LOWER_SLIDE = 0.17
DETECT_DWELL = 1.0
DETECT_MIN_SAMPLES = 4
REACH_FWD_MIN, REACH_FWD_MAX = 0.3, 1.5
REACH_LATERAL_MAX = 0.16
REACH_Z_MIN, REACH_Z_MAX = 0.40, 1.35
GRASP_ROT = np.eye(3)

# scan behaviour
SCAN_DWELL = 1.2
SCAN_SETTLE = 0.8

# phases
(WAIT_TASK, SCAN, NAV_SHELF, DEPLOY, CREEP, CLOSE, LIFT, RETREAT,
 NAV_TABLE, PLACE, NEXT, DONE, ERROR) = range(13)
PHASE_NAME = {
    WAIT_TASK: "wait-task", SCAN: "scan", NAV_SHELF: "nav->shelf",
    DEPLOY: "deploy", CREEP: "creep", CLOSE: "close", LIFT: "lift",
    RETREAT: "retreat", NAV_TABLE: "nav->table", PLACE: "place",
    NEXT: "next", DONE: "done", ERROR: "error",
}


class CompetitionClient(Node):
    def __init__(self):
        super().__init__("competition_client")
        self.adapter = MMK2Adapter()
        self.inventory = ShelfInventory()
        self.task = None
        self.task_listener = TaskListener(on_new_task=self._on_new_task)

        self.products = []
        self.aruco = []
        self.scan_ranges = None
        self.scan_angle_min = 0.0
        self.scan_angle_inc = 0.0

        self.phase = WAIT_TASK
        self.state_t0 = self._now()
        self.target = None            # SlotRecord being attempted
        self.target_kind = None
        self.pending = []             # remaining task kinds (removed on delivery)
        self.failed_slots = set()     # slots that failed this run
        self.phase_timeouts = {
            NAV_SHELF: 90.0, DEPLOY: 20.0, CREEP: 30.0,
            NAV_TABLE: 90.0, PLACE: 20.0,
        }
        self.scan_idx = 0
        self.scan_pitch_idx = 0
        self.scan_route_set = False
        self.nav_idx = 0
        self.nav_mode = "turn"
        self.route = []
        self.route_yaw = GRASP_YAW
        self.det_buf = deque(maxlen=30)
        self.target_locked = False
        self.deploy_world = None
        self.creep_stop_y = None
        self.last_log = 0.0
        self.last_dbg = 0.0
        self.debug = os.environ.get("COMP_DEBUG", "") == "1"
        self.dt = 1.0 / 30.0
        # reactive avoidance / stuck recovery
        self.avoid_dir = 1.0
        self.avoid_until = 0.0
        self.avoid_back_until = 0.0
        self.stuck_pos = None
        self.stuck_t0 = 0.0
        self.stuck_yaw = 0.0
        self.stuck_goal = None
        self.stuck_best_d = None
        self.recover_until = 0.0
        self.deploy_creep_until = 0.0
        self.prev_yaw = None
        self.prev_yaw_t = 0.0
        # grid planner + path follower
        self.planner = GridPlanner()
        self.controller = PurePursuit()
        self.dwb = DWBLocalPlanner()
        self.path = None
        self.path_goal = None
        self.replan_t = 0.0

        self.create_subscription(String, "/competition/product_detections",
                                 self._products_cb, 10)
        self.create_subscription(String, "/competition/aruco_detections",
                                 self._aruco_cb, 10)
        self.create_subscription(LaserScan, "/slamware_ros_sdk_server_node/scan",
                                 self._scan_cb, 10)

        self.create_timer(self.dt, self.tick)
        self.get_logger().info("competition client up")

    # ---- helpers ----
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_new_task(self, task):
        self.task = task
        self.inventory.reset()
        self.pending = [t.kind for t in task.targets]
        self.scan_idx = 0
        self.scan_pitch_idx = 0
        self.scan_route_set = False
        self.get_logger().info(f"new task run={task.run_prefix} kinds={self.pending}")
        self._enter(SCAN)

    def _products_cb(self, msg):
        try:
            self.products = json.loads(msg.data)
        except Exception:  # noqa: BLE001
            self.products = []
        self.inventory.update(self.products, self.aruco)
        if self.debug and self._now() - self.last_dbg > 2.0:
            self.last_dbg = self._now()
            sample = [(p["kind"], p.get("slot"), p.get("slot_source"),
                       [round(x, 2) for x in p["world"]], round(p["conf"], 2))
                      for p in self.products[:6]]
            aids = [(a["id"], [round(x, 2) for x in a["world"]]) for a in self.aruco[:8]]
            self.get_logger().info(f"[dbg] aruco={aids} dets={sample}")

    def _aruco_cb(self, msg):
        try:
            self.aruco = json.loads(msg.data)
        except Exception:  # noqa: BLE001
            self.aruco = []

    def _scan_cb(self, msg):
        self.scan_ranges = np.asarray(msg.ranges, dtype=float)
        self.scan_angle_min = float(msg.angle_min)
        self.scan_angle_inc = float(msg.angle_increment)

    def _enter(self, phase):
        self.phase = phase
        self.state_t0 = self._now()
        self.stuck_goal = None
        self.stuck_best_d = None
        self.stuck_t0 = self._now()

    # ---- navigation ----
    def _set_route(self, route, yaw):
        self.route = [np.asarray(p, dtype=float) for p in route]
        self.route_yaw = yaw
        self.nav_idx = 0
        self.nav_mode = "turn"

    def _follow_route(self):
        a = self.adapter
        if self.nav_idx < len(self.route):
            target = self.route[self.nav_idx]
            delta = target - a.base_xy
            dist = float(np.linalg.norm(delta))
            yaw_err = wrap_to_pi(math.atan2(delta[1], delta[0]) - a.base_yaw)
            if self.nav_mode == "turn":
                a.set_base_velocity(0.0, 1.0 * yaw_err)
                if abs(yaw_err) < 0.10:
                    self.nav_mode = "drive"
            else:
                if dist < 0.15:
                    self.nav_idx += 1
                    self.nav_mode = "turn"
                    a.stop_base()
                elif not self._front_clear():
                    self._avoid_step()
                else:
                    ang = 0.0 if abs(yaw_err) < 0.10 else 1.0 * yaw_err
                    lin = self._approach_speed(dist) * max(0.0, math.cos(yaw_err))
                    a.set_base_velocity(lin, ang)
            return False
        yaw_err = wrap_to_pi(self.route_yaw - a.base_yaw)
        a.set_base_velocity(0.0, 1.0 * yaw_err)
        if abs(yaw_err) < 0.10:
            a.stop_base()
            return True
        return False

    @staticmethod
    def _approach_speed(dist):
        """Decelerate in steps as the waypoint gets close."""
        if dist < 0.40:
            return 0.06
        if dist < 0.80:
            return 0.12
        return 0.25

    # ---- grid-planner navigation ----
    def _navigate(self, goal, final_yaw=None, tol: float = 0.15) -> bool:
        """Plan an A* path to ``goal`` and follow it.  Returns True when reached.

        ``final_yaw`` = None means any final heading is acceptable.
        """
        a = self.adapter
        now = self._now()
        goal = (float(goal[0]), float(goal[1]))
        if self.path_goal != goal:
            self.path = None
            self.path_goal = goal
            self.dwb.reset()
        if self.scan_ranges is not None:
            self.planner.update_scan(
                self.scan_ranges, self.scan_angle_min, self.scan_angle_inc,
                float(a.base_xy[0]), float(a.base_xy[1]), float(a.base_yaw))
        if self.path is None or now - self.replan_t > 1.5:
            self.replan_t = now
            self.path = self.planner.plan(
                (float(a.base_xy[0]), float(a.base_xy[1])), goal)
            if self.path is not None:
                self.get_logger().info(
                    f"planned {len(self.path)} pts -> {np.round(goal, 2)}")

        # stuck recovery: drive away from the nearest obstacle, then replan
        if now < self.recover_until:
            self._command_escape(0.18)
            return False

        # hard safety: never push further if the base centre is inside the
        # inflated obstacles (e.g. drifted into the shelf); back out and replan
        ci, cj = self.planner.world_to_idx(float(a.base_xy[0]), float(a.base_xy[1]))
        if (0 <= ci < self.planner.nx and 0 <= cj < self.planner.ny
                and float(self.planner.margin[ci, cj]) < 0.05):
            self.get_logger().warn("base inside safety buffer; escaping")
            self._command_escape(0.18)
            self.path = None
            self.replan_t = 0.0
            self.recover_until = self._now() + 1.5
            return False

        if self.path is None:
            a.stop_base()
            return False

        # estimate yaw rate for damping
        now2 = self._now()
        if self.prev_yaw is not None and now2 > self.prev_yaw_t:
            yaw_rate = wrap_to_pi(a.base_yaw - self.prev_yaw) / (now2 - self.prev_yaw_t)
            yaw_rate = max(-3.0, min(3.0, yaw_rate))
        else:
            yaw_rate = 0.0
        self.prev_yaw = a.base_yaw
        self.prev_yaw_t = now2

        eff_goal = self.path[-1]
        dist_goal = float(np.hypot(eff_goal[0] - a.base_xy[0],
                                   eff_goal[1] - a.base_xy[1]))
        if dist_goal < tol:
            if final_yaw is None:
                a.stop_base()
                self.path = None
                return True
            yaw_err = wrap_to_pi(final_yaw - a.base_yaw)
            if abs(yaw_err) < 0.10:
                a.stop_base()
                self.path = None
                return True
            ang = 0.6 * yaw_err - 0.4 * yaw_rate
            ang = max(-0.30, min(0.30, ang))
            a.set_base_velocity(0.0, ang)
            return False

        v, w, _ = self.dwb.compute(
            self.planner, (float(a.base_xy[0]), float(a.base_xy[1])),
            a.base_yaw, self.path, eff_goal)
        a.set_base_velocity(v, w)
        return False

    def _front_clear(self):
        """Reactive safety: False if something is within 0.35 m ahead.

        The forward sector is computed from the message's angle_min/increment so
        it is correct regardless of the scan angle convention.
        """
        if self.scan_ranges is None or self.scan_ranges.size == 0:
            return True
        r = self.scan_ranges
        n = r.size
        ang = self.scan_angle_min + self.scan_angle_inc * np.arange(n)
        ang = (ang + math.pi) % (2.0 * math.pi) - math.pi
        mask = np.abs(ang) <= math.radians(50.0)
        front = r[mask]
        front = front[np.isfinite(front) & (front > 0.0)]
        if front.size == 0:
            return True
        return float(np.min(front)) > 0.55

    def _path_ahead(self, base_xy, path, dmin: float, dmax: float, step: float = 0.05):
        """Sample world points along ``path`` at arc distance [dmin, dmax]."""
        pts = []
        prev = np.asarray(base_xy, dtype=float)
        base = prev
        for p in path:
            p = np.asarray(p, dtype=float)
            seg = p - prev
            seglen = float(np.linalg.norm(seg))
            if seglen < 1e-6:
                prev = p
                continue
            n = max(1, int(seglen / step))
            for t in np.linspace(0.0, 1.0, n + 1):
                q = prev + seg * t
                d = float(np.linalg.norm(q - base))
                if dmin <= d <= dmax:
                    pts.append(q)
            prev = p
        return pts

    def _path_blocked(self, half_width: float = 0.28) -> bool:
        """True if a scan point lies inside the robot corridor along the next
        stretch of the planned path (replaces the fixed front-sector stop).

        Self returns inside the footprint box are ignored.
        """
        a = self.adapter
        if (self.scan_ranges is None or self.path is None
                or a.base_xy is None or self.scan_ranges.size == 0):
            return False
        pts = self._path_ahead(a.base_xy, self.path, 0.25, 0.70)
        if not pts:
            return False

        r = self.scan_ranges
        n = r.size
        ang = self.scan_angle_min + self.scan_angle_inc * np.arange(n)
        valid = np.isfinite(r) & (r > 0.05) & (r < 12.0)
        ex_b = r * np.cos(ang) + 0.1137
        ey_b = r * np.sin(ang)
        inside = (np.abs(ex_b) <= 0.35) & (np.abs(ey_b) <= 0.35)
        valid &= ~inside
        if not np.any(valid):
            return False

        lx = a.base_xy[0] + math.cos(a.base_yaw) * 0.1137
        ly = a.base_xy[1] + math.sin(a.base_yaw) * 0.1137
        wang = a.base_yaw + ang
        X = lx + r * np.cos(wang)
        Y = ly + r * np.sin(wang)

        for q in pts:
            d = np.hypot(X - q[0], Y - q[1])
            if np.any(d[valid] < half_width):
                return True
        return False

    def _escape_dir(self):
        """Unit world direction pointing away from the nearest obstacle."""
        a = self.adapter
        i, j = self.planner.world_to_idx(float(a.base_xy[0]), float(a.base_xy[1]))
        if 0 <= i < self.planner.nx and 0 <= j < self.planner.ny:
            di, dj = self.planner.grad_at(i, j)
            n = math.hypot(di, dj)
            if n > 1e-6:
                return di / n, dj / n
        return -math.cos(a.base_yaw), -math.sin(a.base_yaw)

    def _command_escape(self, speed: float = 0.18):
        """Drive away from the nearest obstacle (forward or reverse, whichever is
        closer to the escape direction) with a damped heading correction."""
        a = self.adapter
        ex, ey = self._escape_dir()
        desired = math.atan2(ey, ex)
        err_fwd = wrap_to_pi(desired - a.base_yaw)
        err_rev = wrap_to_pi(desired + math.pi - a.base_yaw)
        if abs(err_fwd) <= abs(err_rev):
            a.set_base_velocity(speed, max(-0.30, min(0.30, 0.8 * err_fwd)))
        else:
            a.set_base_velocity(-speed, max(-0.30, min(0.30, 0.8 * err_rev)))

    def _sector_min(self, a0, a1):
        """Minimum valid range within an angular sector [a0, a1] (rad)."""
        if self.scan_ranges is None or self.scan_ranges.size == 0:
            return float("inf")
        r = self.scan_ranges
        n = r.size
        ang = self.scan_angle_min + self.scan_angle_inc * np.arange(n)
        ang = (ang + math.pi) % (2.0 * math.pi) - math.pi
        mask = (ang >= a0) & (ang <= a1)
        vals = r[mask]
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        return float(np.min(vals)) if vals.size else float("inf")

    def _avoid_step(self):
        """Reactive obstacle avoidance: pick a side and steer around, or back up."""
        a = self.adapter
        now = self._now()
        if now < self.avoid_back_until:
            a.set_base_velocity(-0.15, 0.0)
            return
        if now < self.avoid_until:
            a.set_base_velocity(0.06, 1.0 * self.avoid_dir)
            return
        d_l = self._sector_min(math.radians(20.0), math.radians(85.0))
        d_r = self._sector_min(math.radians(-85.0), math.radians(-20.0))
        if max(d_l, d_r) < 0.30:
            self.avoid_back_until = now + 1.2
            self.avoid_dir = -self.avoid_dir
            a.set_base_velocity(-0.15, 0.0)
            return
        self.avoid_dir = 1.0 if d_l >= d_r else -1.0
        self.avoid_until = now + 1.4
        a.set_base_velocity(0.06, 1.0 * self.avoid_dir)

    # ---- target selection ----
    def _select_target(self):
        for kind in list(self.pending):
            cands = [c for c in self.inventory.candidates(kind)
                     if c.slot not in self.failed_slots]
            if cands:
                self.target = cands[0]
                self.target_kind = kind
                return True
        return False

    def _approach_lane(self):
        s = self.target.slot
        x = SHELF_X[s[0]] + COLUMN_DX[s[2]] - APPROACH_DX
        return [x, YELLOW_MID_Y]

    def _current_goal(self):
        """World goal for the current navigation phase (for progress checks)."""
        if self.phase == SCAN:
            idx = min(self.scan_idx, len(SHELF_X) - 1)
            return (SHELF_X[list(SHELF_X)[idx]], SCAN_Y)
        if self.phase == NAV_SHELF and self.target is not None:
            return tuple(self._approach_lane())
        if self.phase == NAV_TABLE:
            return tuple(TABLE_APPROACH)
        return None

    # ---- perception lock during DEPLOY ----
    def _lock_from_products(self):
        if self.adapter.base_xy is None:
            return False
        for p in self.products:
            if p.get("kind") != self.target_kind:
                continue
            pw = np.asarray(p.get("world"), dtype=float)
            fp = self.adapter.world_to_footprint(pw)
            if fp[0] < REACH_FWD_MIN or fp[0] > REACH_FWD_MAX:
                continue
            if abs(fp[1]) > REACH_LATERAL_MAX:
                continue
            if pw[2] < REACH_Z_MIN or pw[2] > REACH_Z_MAX:
                continue
            self.det_buf.append(pw)
        if len(self.det_buf) < DETECT_MIN_SAMPLES:
            return False
        cand = np.median(np.array(list(self.det_buf)), axis=0)
        self.deploy_world = cand + DEPLOY_OFFSET
        self.creep_stop_y = cand[1] + CREEP_STOP_DY
        return True

    # ---- main tick ----
    def tick(self):
        a = self.adapter
        if not a.ready:
            return

        to = self.phase_timeouts.get(self.phase)
        if to is not None and self._now() - self.state_t0 > to:
            self._on_timeout()
            a.step()
            self._log()
            return

        if self.phase in (SCAN, NAV_SHELF, NAV_TABLE) and a.base_xy is not None:
            goal = self._current_goal()
            if goal is not None:
                if self.stuck_goal != goal:
                    self.stuck_goal = goal
                    self.stuck_best_d = None
                    self.stuck_t0 = self._now()
                d = float(np.hypot(goal[0] - a.base_xy[0], goal[1] - a.base_xy[1]))
                if d < 0.20:
                    # at the goal (e.g. scanning): not stuck
                    self.stuck_best_d = d
                    self.stuck_t0 = self._now()
                elif self.stuck_best_d is None or d < self.stuck_best_d - 0.15:
                    self.stuck_best_d = d
                    self.stuck_t0 = self._now()
                elif self._now() - self.stuck_t0 > 6.0:
                    self.get_logger().warn(
                        f"stuck (no progress toward {np.round(goal, 2)}, d={d:.2f}); "
                        f"backing out + replan")
                    self.path = None
                    self.replan_t = 0.0
                    self.recover_until = self._now() + 3.0
                    self.stuck_best_d = d
                    self.stuck_t0 = self._now()

        if self.phase == WAIT_TASK:
            a.stop_base()
        elif self.phase == SCAN:
            self._tick_scan()
        elif self.phase == NAV_SHELF:
            if self._navigate(self._approach_lane(), GRASP_YAW, tol=0.06):
                self.det_buf.clear()
                self.target_locked = False
                self._enter(DEPLOY)
        elif self.phase == DEPLOY:
            a.stop_base()
            a.set_head(0.0, HEAD_PITCH)
            a.set_slide(SLIDE_GRASP)
            a.set_gripper("right", GRIP_OPEN)
            if not self.target_locked and self._now() - self.state_t0 > DETECT_DWELL:
                if self._lock_from_products():
                    if a.arm_to("right", self.deploy_world, GRASP_ROT):
                        self.target_locked = True
                        self.get_logger().info(
                            f"locked {self.target_kind} world={np.round(self.deploy_world, 3)}")
                        self._enter(CREEP)
                    else:
                        self.get_logger().warn(
                            f"IK failed for {np.round(self.deploy_world, 3)}, retrying")
                        self.det_buf.clear()
        elif self.phase == CREEP:
            ee = a.ee_world("right")
            if ee[1] < self.creep_stop_y:
                a.set_base_velocity(CREEP_SPEED, 1.5 * wrap_to_pi(GRASP_YAW - a.base_yaw))
            else:
                a.stop_base()
                self._enter(CLOSE)
        elif self.phase == CLOSE:
            a.stop_base()
            a.set_gripper("right", GRIP_CLOSE)
            if self._now() - self.state_t0 > 0.8:
                self._enter(LIFT)
        elif self.phase == LIFT:
            a.stop_base()
            a.set_slide(SLIDE_GRASP - LIFT_AMOUNT)
            if abs(a.slide_meas - (SLIDE_GRASP - LIFT_AMOUNT)) < 0.02:
                self._enter(RETREAT)
        elif self.phase == RETREAT:
            yaw_err = wrap_to_pi(GRASP_YAW - a.base_yaw)
            if a.base_xy[1] > YELLOW_MID_Y + 0.06:
                a.set_base_velocity(-RETREAT_SPEED, 1.0 * yaw_err)
            else:
                a.stop_base()
                self._enter(NAV_TABLE)
        elif self.phase == NAV_TABLE:
            if self._navigate(TABLE_APPROACH, DELIVERY_YAW):
                self._enter(PLACE)
        elif self.phase == PLACE:
            a.stop_base()
            a.set_slide(PLACE_LOWER_SLIDE)
            if abs(a.slide_meas - PLACE_LOWER_SLIDE) < 0.03:
                a.set_gripper("right", GRIP_OPEN)
                if self._now() - self.state_t0 > 1.0:
                    self._enter(NEXT)
        elif self.phase == NEXT:
            if self.target_kind in self.pending:
                self.pending.remove(self.target_kind)
            self.target = None
            self.target_kind = None
            self.target_locked = False
            self.det_buf.clear()
            self._recover()
        elif self.phase == DONE:
            a.stop_base()
        else:
            a.stop_base()

        a.step()
        self._log()

    def _tick_scan(self):
        a = self.adapter
        # if a pending target is already known, go straight to it
        if self._select_target():
            self._start_nav_shelf()
            return
        if self.scan_idx >= len(SHELF_X):
            self._enter(ERROR if self.pending else DONE)
            return
        shelf = list(SHELF_X)[self.scan_idx]
        obs = [SHELF_X[shelf], SCAN_Y]
        if a.base_xy is None:
            a.stop_base()
            return
        if not self._navigate(obs, None):
            return
        a.stop_base()
        a.set_slide(SCAN_SLIDE)
        pitch = SCAN_PITCHES[self.scan_pitch_idx]
        a.set_head(0.0, pitch)
        if self._now() - self.state_t0 > SCAN_DWELL:
            self.scan_pitch_idx += 1
            self.state_t0 = self._now()
            if self.scan_pitch_idx >= len(SCAN_PITCHES):
                self.scan_pitch_idx = 0
                self.scan_idx += 1
                self.scan_route_set = False

    def _start_nav_shelf(self):
        lane = self._approach_lane()
        self.path = None
        self.path_goal = None
        self.get_logger().info(
            f"target kind={self.target_kind} slot={self.target.slot} lane={lane}")
        self._enter(NAV_SHELF)

    def _on_timeout(self):
        phase = self.phase
        self.get_logger().warn(
            f"timeout in {PHASE_NAME[phase]} "
            f"target={self.target.slot if self.target else None}")
        self.adapter.stop_base()
        if phase in (NAV_SHELF, DEPLOY, CREEP):
            self.adapter.home()
            if self.target is not None:
                self.failed_slots.add(self.target.slot)
            self.target = None
            self.target_locked = False
            self.det_buf.clear()
            self._recover()
        else:
            # holding or placing: do NOT home (avoid dropping); stop and finish
            self._enter(DONE)

    def _recover(self):
        if self.pending and self._select_target():
            self._start_nav_shelf()
        elif self.pending:
            self.scan_idx = 0
            self.scan_pitch_idx = 0
            self.scan_route_set = False
            self._enter(SCAN)
        else:
            self._enter(DONE)

    def _log(self):
        if self._now() - self.last_log < 1.0:
            return
        self.last_log = self._now()
        a = self.adapter
        inv = self.inventory.summary()
        self.get_logger().info(
            f"phase={PHASE_NAME[self.phase]} base=({a.base_xy[0]:.2f},{a.base_xy[1]:.2f}) "
            f"yaw={a.base_yaw:.2f} cmd=({a.des_lin:.2f},{a.des_ang:.2f}) "
            f"front_clear={self._front_clear()} nav={self.nav_idx}/{len(self.route)}:{self.nav_mode} "
            f"pending={self.pending} inv={inv}")


def main():
    rclpy.init()
    client = CompetitionClient()
    executor = SingleThreadedExecutor()
    executor.add_node(client)
    executor.add_node(client.adapter)
    executor.add_node(client.task_listener)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            client.adapter.emergency_stop()
        except Exception:  # noqa: BLE001
            pass
        executor.shutdown()
        for n in (client, client.adapter, client.task_listener):
            n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
