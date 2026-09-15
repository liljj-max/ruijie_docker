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
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from competition_client.controller import PurePursuit
from competition_client.grasp_logic import StableTargetTracker, creep_speed
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
_SHELF_ORDER = ["A", "B", "C", "D", "E"]
_LEVEL_ORDER = ["L1", "L2", "L3"]
_COLUMN_ORDER = ["C1", "C2", "C3"]


def slot_to_aruco_id(slot):
    """Map a (shelf, level, column) slot to its fixed ArUco marker id."""
    try:
        shelf, level, column = slot
        return (_SHELF_ORDER.index(shelf) * 9 + _LEVEL_ORDER.index(level) * 3
                + _COLUMN_ORDER.index(column))
    except (ValueError, TypeError):
        return None
SCAN_Y = 2.40                # observation lane in front of the shelves (safer)
SCAN_SLIDE = 0.11            # raise the head to shelf level while scanning
SCAN_YAWS = [0.0, -0.15, 0.15]          # level gaze first, then a small deflection
SCAN_PITCHES = [-0.30, -0.70, -1.10]    # L3, L2, L1 (top shelf first)
TABLE_APPROACH = [-1.88, -2.80]
# Fixed entry to the obstacle zone: the midpoint of the shelf-zone / obstacle-
# zone boundary (dynamic_obstacle_zone_north_mark is at world y=1.70, x spans
# [-2.42, 0.50]).  Entering from the same pose every time keeps the A* start and
# heading consistent, instead of a different pose each run.
OBSTACLE_ENTRY = [-0.50, YELLOW_MID_Y]   # north of the corridor board; avoidance starts here

# manipulation params (from the reference baseline)
HEAD_PITCH = -0.6
# The deploy pose must keep the target in the head camera's view.  The scan
# pitch per level was calibrated at SCAN_SLIDE, but the deploy uses a different
# spine slide (grasp_slide), which raises/lowers the whole head by
# (grasp_slide - SCAN_SLIDE) metres (spine z = 1.406 - q).  Compensate the
# pitch by that height difference at ~0.77 m range.
SCAN_PITCH_BY_LEVEL = {"L3": -0.30, "L2": -0.70, "L1": -1.10}
DEPLOY_PITCH_K = 1.25
# The scan looks at the shelf with the base at pi/2 and head yaw 0, but the
# grasp approach turns the base to GRASP_YAW = pi/2 - 11 deg.  The deploy head
# yaw must compensate, otherwise the (top-shelf) target drops out of the
# camera's view and can never be locked.
DEPLOY_HEAD_YAW = wrap_to_pi(math.pi / 2.0 - GRASP_YAW)
# If the target is still not seen, sweep (dyaw, dpitch) around the level pitch
# to re-acquire it before giving up.
DEPLOY_LOOK = [(0.0, 0.0), (0.0, 0.15), (0.0, -0.15),
               (-0.20, 0.0), (0.20, 0.0)]
DEPLOY_PITCH_DWELL = 1.5   # s to hold each look before trying the next
SLIDE_GRASP = 0.11
# The spine (slide) raises/lowers the chest, so the reachable height depends on
# it.  L1 sits below the reach envelope at SLIDE_GRASP, so lower the chest more
# for the lower shelves.
SLIDE_GRASP_BY_LEVEL = {"L1": 0.45, "L2": 0.11, "L3": 0.30}
LIFT_AMOUNT = 0.05
DEPLOY_OFFSET = np.array([-0.011, -0.220, -0.010])
MIN_DEPLOY_FWD = 0.58        # IK has a reach hole closer than ~0.55 m at shelf height
CREEP_STOP_GAP = 0.035
RETREAT_SPEED = 0.12
PLACE_LOWER_SLIDE = 0.17
DETECT_DWELL = 1.0
GRASP_ROT = np.eye(3)

# Placement on the fixed delivery table (world frame).  The base faces south at
# TABLE_APPROACH; the right arm reaches forward over the table top, lowers to
# just above the surface, releases, then lifts before homing.  The target is a
# little forward of the table centre so the forward reach stays inside the IK
# envelope (there is a reach hole closer than ~0.55 m).
TABLE_PLACE_XY = np.array([-1.94, -3.45])  # further over the table centre
TABLE_TOP_Z = 0.767
ITEM_HALF_H = 0.0725                       # kele cylinder half-height
PLACE_HOVER_Z = TABLE_TOP_Z + ITEM_HALF_H + 0.06    # ~0.90: hover above the table
PLACE_DROP_Z = TABLE_TOP_Z + ITEM_HALF_H - 0.005    # ~0.835: set down on the surface
PLACE_RAISE_Z = 1.00                       # lift clear before homing
PLACE_RELEASE_DWELL = 1.3                  # s to wait after opening before lifting

# scan behaviour
SCAN_DWELL = 0.5          # base dwell per view
SCAN_DWELL_MAX = 1.2      # extend while new detections keep arriving
SCAN_SETTLE = 0.4

# hard-coded exploration primitives (departure + shelf zone)
EXPLORE_SPEED = 0.10          # straight-line cruise (departure / RETURN primitives)
EXPLORE_DRIVE_TOL = 0.06
EXPLORE_TURN_TOL = 0.03
EXPLORE_TURN_MAX = 0.15       # turn rate cap (a19bba0 value; avoids RETURN swaying)
# The shelf-scan plan may move/turn faster than the RETURN/ALIGN primitives
# (it is a straight lane run with no held item), so scope the speed-up to SCAN.
SCAN_SPEED = 0.15
SCAN_TURN_MAX = 0.25
# Final in-place alignment before grasping can spin a bit faster than the
# travel turns (no translation, so the RETURN swaying is not a concern).
ALIGN_TURN_MAX = 0.22
ALIGN_TURN_GAIN = 0.7

# phases
(WAIT_TASK, STOW, SCAN, ALIGN, DEPLOY, WAIT_ARM, CREEP, BRAKE, CLOSE, LIFT,
 RETREAT, RETURN, NAV_TABLE, PLACE, NEXT, NAV_RETURN, DONE, ERROR) = range(18)
PHASE_NAME = {
    WAIT_TASK: "wait-task", STOW: "stow", SCAN: "scan", ALIGN: "align",
    DEPLOY: "deploy", WAIT_ARM: "wait-arm", CREEP: "creep", BRAKE: "brake",
    CLOSE: "close", LIFT: "lift",
    RETREAT: "retreat", RETURN: "return", NAV_TABLE: "nav->table", PLACE: "place",
    NEXT: "next", NAV_RETURN: "nav->shelf-zone", DONE: "done", ERROR: "error",
}


class CompetitionClient(Node):
    def __init__(self):
        super().__init__("competition_client")
        self.adapter = MMK2Adapter()
        self.inventory = ShelfInventory()
        self.task = None
        self.task_listener = TaskListener(on_new_task=self._on_new_task)

        self.products = []
        self.products_seq = 0
        self.aruco = []
        self.handeye_aruco = []
        self.handeye_rx_t = 0.0
        self.fine_adjusted = False
        self.scan_ranges = None
        self.scan_angle_min = 0.0
        self.scan_angle_inc = 0.0
        self.scan_range_min = 0.02
        self.scan_range_max = 12.0
        self.scan_frame = "laser"
        self.scan_rx_t = 0.0
        self.scan_stamp = 0.0
        self.scan_time = Time()
        self.scan_seq = 0
        self.mapped_scan_seq = -1

        self.phase = WAIT_TASK
        self.state_t0 = self._now()
        self.target = None            # SlotRecord being attempted
        self.target_order = None      # exact TaskTarget; id is identity only
        self.target_kind = None
        self.pending = []             # remaining TaskTarget objects
        self.failed_slots = set()     # slots that failed this run
        # Generous per-phase timeouts: the base is slow, so give each phase room
        # to finish rather than aborting mid-route (stability over speed).
        self.phase_timeouts = {
            STOW: 30.0, ALIGN: 90.0, RETURN: 200.0, NAV_RETURN: 250.0,
            DEPLOY: 30.0,
            WAIT_ARM: 30.0, CREEP: 40.0, BRAKE: 5.0, CLOSE: 6.0,
            LIFT: 15.0, RETREAT: 30.0, NAV_TABLE: 300.0, PLACE: 40.0,
        }
        self.scan_yaw_idx = 0
        self.scan_pitch_idx = 0
        self.view_t0 = 0.0
        self.view_start = 0.0
        self.view_settle_t0 = 0.0
        self.view_products_seq = -1
        self.view_inv = 0
        self.align_stage = "pos"
        self.align_settle_t0 = 0.0
        self.place_sub = 0
        self.place_sub_t0 = 0.0
        self.recover_scans = 0
        self.deploy_pitch_idx = 0
        self.deploy_pitch_t = 0.0
        self.explore_plan = None
        self.explore_i = 0
        self.scan_complete = False
        self.scan_views = 0
        self.scanned_shelves = set()
        self.prim_start_xy = None
        self.prim_start_yaw = 0.0
        self.nav_idx = 0
        self.nav_mode = "turn"
        self.route = []
        self.route_yaw = GRASP_YAW
        self.target_tracker = StableTargetTracker()
        self.target_locked = False
        self.deploy_world = None
        self.grasp_world = None
        self.motion_settle_t0 = 0.0
        self.grasp_slide = SLIDE_GRASP
        self.stall_ref_xy = None
        self.stall_ref_t = 0.0
        self.reverse_until = 0.0
        self.stall_count = 0
        self.stall_spot = None
        self.last_arm_dbg = 0.0
        self.last_nav_dbg = 0.0
        self.stow_next = SCAN
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
        self.last_sensor_warn = 0.0
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(String, "/competition/product_detections",
                                 self._products_cb, 10)
        self.create_subscription(String, "/competition/aruco_detections",
                                 self._aruco_cb, 10)
        self.create_subscription(String, "/competition/handeye_aruco_detections",
                                 self._handeye_cb, 10)
        self.create_subscription(LaserScan, "/slamware_ros_sdk_server_node/scan",
                                 self._scan_cb, 10)

        self.create_timer(self.dt, self.tick)
        self.get_logger().info("competition client up")

    # ---- helpers ----
    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_new_task(self, task):
        holding = self.phase in (LIFT, RETREAT, RETURN, NAV_TABLE, PLACE)
        if holding and self.adapter.gripper_meas("right") < 0.5:
            self.adapter.stop_base()
            self.get_logger().error(
                "new run received while carrying an item; stopping without opening gripper")
            self._enter(ERROR)
            return
        self.task = task
        self.adapter.stop_base()
        self.inventory.reset()
        self.pending = list(task.targets)
        self.failed_slots.clear()
        self.products = []
        self.aruco = []
        self.target = None
        self.target_order = None
        self.target_kind = None
        self.target_locked = False
        self.deploy_world = None
        self.grasp_world = None
        self.scan_yaw_idx = 0
        self.scan_pitch_idx = 0
        self.explore_plan = None
        self.explore_i = 0
        self.scan_complete = False
        self.scan_views = 0
        self.scanned_shelves = set()
        self.recover_scans = 0
        self.prim_start_xy = None
        self.target_tracker.clear()
        self.path = None
        self.path_goal = None
        kinds = [t.kind for t in self.pending]
        self.get_logger().info(f"new task run={task.run_prefix} kinds={kinds}")
        self.stow_next = SCAN
        self._enter(STOW)

    def _products_cb(self, msg):
        try:
            self.products = json.loads(msg.data)
        except Exception:  # noqa: BLE001
            self.products = []
        self.products_seq += 1
        scanning_view = (
            self.phase == SCAN
            and self.explore_plan is not None
            and self.explore_i < len(self.explore_plan)
            and self.explore_plan[self.explore_i][0] == "scan"
            and self.view_settle_t0 > 0.0
            and self._now() - self.view_settle_t0 >= SCAN_SETTLE
        )
        if scanning_view:
            self.inventory.update(self.products)
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

    def _handeye_cb(self, msg):
        try:
            self.handeye_aruco = json.loads(msg.data)
        except Exception:  # noqa: BLE001
            self.handeye_aruco = []
        self.handeye_rx_t = self._now()

    def _handeye_marker_world(self, slot, max_age: float = 0.5):
        """Latest fresh hand-eye world position of ``slot``'s ArUco marker."""
        marker_id = slot_to_aruco_id(slot)
        if marker_id is None or self._now() - self.handeye_rx_t > max_age:
            return None
        for det in self.handeye_aruco:
            if int(det.get("id", -1)) == marker_id:
                world = det.get("world")
                if world is not None and len(world) >= 3:
                    return np.asarray(world, dtype=float)
        return None

    def _scan_cb(self, msg):
        self.scan_ranges = np.asarray(msg.ranges, dtype=float)
        self.scan_angle_min = float(msg.angle_min)
        self.scan_angle_inc = float(msg.angle_increment)
        self.scan_range_min = float(msg.range_min)
        self.scan_range_max = float(msg.range_max)
        self.scan_frame = msg.header.frame_id.lstrip("/") or "laser"
        self.scan_rx_t = self._now()
        self.scan_stamp = float(msg.header.stamp.sec) + msg.header.stamp.nanosec * 1e-9
        if self.scan_stamp <= 0.0:
            self.scan_stamp = self.scan_rx_t
        self.scan_time = Time.from_msg(msg.header.stamp)
        self.scan_seq += 1

    def _enter(self, phase):
        self.phase = phase
        self.state_t0 = self._now()
        self.stuck_goal = None
        self.stuck_best_d = None
        self.stuck_t0 = self._now()
        if phase == PLACE:
            self.place_sub = 0
            self.place_sub_t0 = self._now()
        elif phase == DEPLOY:
            self.deploy_pitch_idx = 0
            self.deploy_pitch_t = self._now()

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
    def _laser_pose_in_base(self):
        """Return live base_link->laser planar TF, or None when unavailable."""
        if self.scan_frame == "base_link":
            return 0.0, 0.0, 0.0
        try:
            tf = self.tf_buffer.lookup_transform(
                "base_link", self.scan_frame, self.scan_time)
        except Exception as exc:  # noqa: BLE001
            if self._now() - self.last_sensor_warn > 2.0:
                self.last_sensor_warn = self._now()
                self.get_logger().warn(
                    f"missing base_link->{self.scan_frame} TF: {exc}")
            return None
        t = tf.transform.translation
        q = tf.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return float(t.x), float(t.y), float(yaw)

    def _planned_path_invalid(self) -> bool:
        if self.path is None or self.adapter.base_xy is None:
            return False
        points = self._path_ahead(self.adapter.base_xy, self.path, 0.10, 1.00)
        for x, y in points:
            i, j = self.planner.world_to_idx(float(x), float(y))
            if (not (0 <= i < self.planner.nx and 0 <= j < self.planner.ny)
                    or self.planner.blocked[i, j]):
                return True
        return False

    def _nav_abort(self, reason: str):
        if self._now() - self.last_nav_dbg > 1.0:
            self.last_nav_dbg = self._now()
            self.get_logger().warn(f"[nav-abort] {reason}")

    def _navigate(self, goal, final_yaw=None, tol: float = 0.15) -> bool:
        """Plan an A* path to ``goal`` and follow it.  Returns True when reached.

        ``final_yaw`` = None means any final heading is acceptable.
        """
        a = self.adapter
        # Compliance guard: avoidance planning is ONLY for the obstacle zone.
        # The departure / shelf / pick zones must use the hard-coded primitives.
        if self.phase not in (NAV_TABLE, NAV_RETURN):
            a.stop_base()
            return False
        now = self._now()
        goal = (float(goal[0]), float(goal[1]))
        if self.path_goal != goal:
            self.path = None
            self.path_goal = goal
            self.dwb.reset()
        if self.scan_ranges is None or now - self.scan_rx_t > 1.5:
            a.stop_base()
            self.path = None
            self._nav_abort(
                f"scan stale (age={now - self.scan_rx_t:.2f}s)")
            return False
        if self.mapped_scan_seq != self.scan_seq:
            sensor_pose = self._laser_pose_in_base()
            odom_pose = self.adapter.pose_at(self.scan_stamp)
            if odom_pose is None:
                # Fall back to the newest odom (<=~0.2 s stale, ~1 cm) instead
                # of stopping: a scan frame without a time-aligned sample must
                # not freeze navigation.
                odom_pose = self.adapter.latest_pose()
            if sensor_pose is None or odom_pose is None:
                a.stop_base()
                self.path = None
                self._nav_abort(
                    f"no pose (sensor={sensor_pose is not None} "
                    f"odom={odom_pose is not None})")
                return False
            scan_xy, scan_yaw = odom_pose
            self.planner.update_scan(
                self.scan_ranges, self.scan_angle_min, self.scan_angle_inc,
                float(scan_xy[0]), float(scan_xy[1]), float(scan_yaw),
                sensor_pose=sensor_pose,
                range_min=self.scan_range_min, range_max=self.scan_range_max)
            self.mapped_scan_seq = self.scan_seq
            if self._planned_path_invalid():
                self.get_logger().info("planned path blocked; replanning")
                self.path = None
                self.replan_t = 0.0
        if self.path is None or now - self.replan_t > 1.0:
            self.replan_t = now
            self.path = self.planner.plan(
                (float(a.base_xy[0]), float(a.base_xy[1])), goal)
            if self.path is not None:
                if math.hypot(self.path[-1][0] - goal[0],
                              self.path[-1][1] - goal[1]) > 0.25:
                    self.get_logger().warn("planner moved goal too far; stopping")
                    self.path = None
                    a.stop_base()
                    return False
                self.get_logger().info(
                    f"planned {len(self.path)} pts -> {np.round(goal, 2)}")

        # Never issue an unvalidated escape command inside the safety buffer.
        ci, cj = self.planner.world_to_idx(float(a.base_xy[0]), float(a.base_xy[1]))
        # Trigger the recovery as soon as the DWB's own safety margin is
        # violated; otherwise the robot deadlocks in the gap between the escape
        # threshold and the DWB safety (no feasible sample, no escape).
        if (0 <= ci < self.planner.nx and 0 <= cj < self.planner.ny
                and float(self.planner.margin[ci, cj]) < self.dwb.safety + 0.01):
            if self._now() - self.last_sensor_warn > 1.0:
                self.last_sensor_warn = self._now()
                self.get_logger().warn("base inside safety buffer; escaping")
            self._escape_step()
            self.path = None
            self.replan_t = 0.0
            return False

        if self.path is None:
            a.stop_base()
            self._nav_abort("planner returned no path")
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
            ang = max(-0.18, min(0.18, ang))
            a.set_base_velocity(0.0, ang)
            return False

        v, w, _ = self.dwb.compute(
            self.planner, (float(a.base_xy[0]), float(a.base_xy[1])),
            a.base_yaw, self.path, eff_goal)
        # Deadlock: every forward arc collides and "do nothing" wins the score,
        # so the DWB returns (0, 0).  Neither the stall recovery (needs a
        # nonzero command) nor the buffer escape (needs a low margin) fires, so
        # the robot sits until the phase times out.  Break it explicitly.
        if abs(v) < 1e-3 and abs(w) < 1e-3:
            if self._now() - self.last_sensor_warn > 1.0:
                self.last_sensor_warn = self._now()
                self.get_logger().warn(
                    "[nav] local planner deadlock; escaping/rotating")
            if not self._escape_step():
                tgt = self.path[1] if len(self.path) >= 2 else eff_goal
                heading = math.atan2(tgt[1] - a.base_xy[1],
                                     tgt[0] - a.base_xy[0])
                a.set_base_velocity(
                    0.0, max(-0.18, min(0.18,
                                        0.8 * wrap_to_pi(heading - a.base_yaw))))
            self.path = None
            self.replan_t = 0.0
            return False
        if self._stall_recovery():
            return False
        if self._now() - self.last_nav_dbg > 1.0:
            self.last_nav_dbg = self._now()
            self.get_logger().info(
                f"[nav] path={[tuple(np.round(p, 2)) for p in self.path[:6]]} "
                f"margin={float(self.planner.margin[ci, cj]):.3f} "
                f"v={v:.3f} w={w:.3f} goal={np.round(eff_goal, 2)}")
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

    def _command_escape(self, speed: float = 0.12):
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

    def _laser_min_in_dir(self, bearing: float, half: float = math.radians(30.0)):
        """Minimum valid LaserScan range around a base-frame bearing."""
        if self.scan_ranges is None or self.scan_ranges.size == 0:
            return float("inf")
        r = self.scan_ranges
        ang = self.scan_angle_min + self.scan_angle_inc * np.arange(r.size)
        sensor = self._laser_pose_in_base()
        syaw = sensor[2] if sensor is not None else 0.0
        rel = (ang + syaw - bearing + math.pi) % (2.0 * math.pi) - math.pi
        vals = r[np.abs(rel) <= half]
        vals = vals[np.isfinite(vals) & (vals > 0.05)]
        return float(np.min(vals)) if vals.size else float("inf")

    def _stall_recovery(self) -> bool:
        """Detect a physically stalled base (commanded but the base pose does not
        change) and back straight out for a moment.

        Uses the odom POSE delta, not the odom twist (the simulator reports a
        near-zero twist even while the base moves).  Corridor only.
        """
        a = self.adapter
        now = self._now()
        if a.base_xy is None:
            return False
        commanded = abs(a.des_lin) > 0.03 or abs(a.des_ang) > 0.05
        if self.stall_ref_xy is None:
            self.stall_ref_xy = a.base_xy.copy()
            self.stall_ref_t = now
        moved = float(np.linalg.norm(a.base_xy - self.stall_ref_xy))
        if moved > 0.03 or not commanded:
            # Real progress away from the last wedge resets the escalation.
            if (self.stall_spot is not None
                    and float(np.linalg.norm(a.base_xy - self.stall_spot)) > 0.40):
                self.stall_count = 0
                self.stall_spot = None
            self.stall_ref_xy = a.base_xy.copy()
            self.stall_ref_t = now
        elif now - self.stall_ref_t > 2.0:
            # Repeated stalls near the same spot: back out progressively further
            # so the base actually clears the wedge instead of nosing straight
            # back into it.
            self.stall_count += 1
            self.stall_spot = a.base_xy.copy()
            back = 1.5 + min(2.5, 0.8 * (self.stall_count - 1))
            self.reverse_until = now + back
            self.stall_ref_xy = a.base_xy.copy()
            self.stall_ref_t = now
            self.get_logger().warn(
                f"[nav] base stalled (no pose change); reversing {back:.1f}s")
        if now < self.reverse_until:
            if self._laser_min_in_dir(a.base_yaw + math.pi) > 0.20:
                a.set_base_velocity(-0.08, 0.0)
            else:
                a.stop_base()
            return True
        return False

    def _escape_step(self, speed: float = 0.06) -> bool:
        """Validated recovery from the inflation buffer: drive away from the
        nearest obstacle, but only in a direction the LaserScan says is clear."""
        a = self.adapter
        ex, ey = self._escape_dir()
        desired = math.atan2(ey, ex)
        err_fwd = wrap_to_pi(desired - a.base_yaw)
        err_rev = wrap_to_pi(desired + math.pi - a.base_yaw)
        if abs(err_fwd) <= abs(err_rev):
            if self._laser_min_in_dir(a.base_yaw) > 0.18:
                a.set_base_velocity(speed, max(-0.15, min(0.15, 0.8 * err_fwd)))
                return True
        elif self._laser_min_in_dir(a.base_yaw + math.pi) > 0.18:
            a.set_base_velocity(-speed, max(-0.15, min(0.15, 0.8 * err_rev)))
            return True
        a.stop_base()
        return False

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
        """Reserve the best visually observed candidate for a pending order."""
        bx = float(self.adapter.base_xy[0]) if self.adapter.base_xy is not None else None
        best = None
        for order in self.pending:
            for c in self.inventory.candidates(order.kind):
                if c.slot in self.failed_slots:
                    continue
                score = c.confidence
                if bx is not None:
                    sx = SHELF_X.get(c.slot[0])
                    if sx is not None:
                        score -= 0.5 * abs(sx - bx)   # prefer the current shelf
                if best is None or score > best[0]:
                    best = (score, c, order)
        if best is None:
            return False
        if not self.inventory.reserve(best[1].slot):
            return False
        self.target = best[1]
        self.target_order = best[2]
        self.target_kind = best[2].kind
        return True

    def _approach_lane(self):
        s = self.target.slot
        x = SHELF_X[s[0]] + COLUMN_DX[s[2]] - APPROACH_DX
        return [x, YELLOW_MID_Y]

    def _current_goal(self):
        """World goal for the current navigation phase (for progress checks)."""
        if self.phase == SCAN:
            return None  # primitive-driven; no progress check needed
        if self.phase == NAV_TABLE:
            return tuple(TABLE_APPROACH)
        if self.phase == NAV_RETURN:
            return tuple(OBSTACLE_ENTRY)
        return None

    # ---- perception lock during DEPLOY ----
    def _lock_from_products(self):
        if self.adapter.base_xy is None:
            return False
        stamps = [p.get("stamp") for p in self.products if p.get("stamp") is not None]
        if not stamps:
            return False
        cand = self.target_tracker.add_frame(
            max(stamps), self.products, self.target_kind, self.target.slot,
            self.adapter.world_to_footprint, self.target.aruco_id)
        if cand is None:
            return False
        self.grasp_world = cand
        self.deploy_world = cand + DEPLOY_OFFSET
        # keep the deploy pose out of the arm's IK reach hole near the chest
        fp = self.adapter.world_to_footprint(self.deploy_world)
        if fp[0] < MIN_DEPLOY_FWD:
            fp[0] = MIN_DEPLOY_FWD
            self.deploy_world = self.adapter.footprint_to_world(fp)
        self.fine_adjusted = False
        return True

    def _fine_adjust_target(self):
        """Refine the grasp x/y once with the hand-eye ArUco marker of the slot.

        The head camera localises the product at ~0.6-0.9 m; the eye-in-hand
        camera sees the slot marker up close, so its world x/y is used to
        correct the deploy target while the arm is already at the deploy pose.
        """
        if self.fine_adjusted or self.target is None:
            return False
        marker = self._handeye_marker_world(self.target.slot)
        if marker is None:
            return False
        self.fine_adjusted = True
        old_grasp = self.grasp_world.copy()
        old_deploy = self.deploy_world.copy()
        dx = float(marker[0] - old_grasp[0])
        dy = float(marker[1] - old_grasp[1])
        self.grasp_world = np.array([marker[0], marker[1], old_grasp[2]])
        self.deploy_world = self.grasp_world + DEPLOY_OFFSET
        fp = self.adapter.world_to_footprint(self.deploy_world)
        if fp[0] < MIN_DEPLOY_FWD:
            fp[0] = MIN_DEPLOY_FWD
            self.deploy_world = self.adapter.footprint_to_world(fp)
        if not self.adapter.arm_to("right", self.deploy_world, GRASP_ROT):
            self.grasp_world = old_grasp
            self.deploy_world = old_deploy
            self.get_logger().warn("[fine] marker seen but IK failed; keep head target")
            return False
        self.get_logger().info(
            f"[fine] marker id={slot_to_aruco_id(self.target.slot)} "
            f"world={np.round(marker, 3)} dx={dx:+.3f} dy={dy:+.3f}")
        return True

    # ---- main tick ----
    def tick(self):
        a = self.adapter
        self._tick_t0 = self._now()
        if not a.ready:
            a.emergency_stop()
            return

        to = self.phase_timeouts.get(self.phase)
        if to is not None and self._now() - self.state_t0 > to:
            self._on_timeout()
            a.step()
            self._log()
            return

        if self.phase in (SCAN, NAV_TABLE, NAV_RETURN) and a.base_xy is not None:
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
                        f"stopping + replan")
                    self.path = None
                    self.replan_t = 0.0
                    a.stop_base()
                    self.stuck_best_d = d
                    self.stuck_t0 = self._now()

        if self.phase == WAIT_TASK:
            a.stop_base()
        elif self.phase == STOW:
            a.stop_base()
            a.home()
            arm_ok = a.arm_settled("right", pos_tol=0.08, vel_tol=0.10)
            slide_ok = abs(a.slide_meas) < 0.03
            grip_ok = a.gripper_settled("right", GRIP_OPEN,
                                        pos_tol=0.08, vel_tol=0.10)
            if self._now() - self.last_arm_dbg > 0.5:
                self.last_arm_dbg = self._now()
                per = np.abs(a.tc[12:18] - a.arm_meas("right"))
                pos_err, max_vel = a.arm_errors("right")
                self.get_logger().info(
                    f"[stow] arm_ok={arm_ok} slide_ok={slide_ok}"
                    f"({a.slide_meas:.3f}) grip_ok={grip_ok}"
                    f"({a.gripper_meas('right'):.3f}) "
                    f"pos_err={pos_err:.3f} max_vel={max_vel:.3f} "
                    f"per={np.round(per, 3)} jidx={int(np.argmax(per)) + 1}")
            settled = arm_ok and slide_ok and grip_ok
            if settled:
                if self.motion_settle_t0 == 0.0:
                    self.motion_settle_t0 = self._now()
                elif self._now() - self.motion_settle_t0 >= 0.3:
                    self.motion_settle_t0 = 0.0
                    if self.stow_next == SCAN:
                        self._enter(SCAN)
                    elif self.stow_next == NAV_RETURN:
                        self.path = None
                        self.path_goal = None
                        self._enter(NAV_RETURN)
                    elif self.stow_next == ALIGN:
                        self._start_nav_shelf()
                    else:
                        self._enter(self.stow_next)
            else:
                self.motion_settle_t0 = 0.0
        elif self.phase == SCAN:
            self._tick_scan()
        elif self.phase == ALIGN:
            lane = self._approach_lane()
            if self.align_stage == "pos":
                if self._drive_to(lane[0], lane[1]):
                    self.align_stage = "yaw"
                    self.align_settle_t0 = 0.0
            else:
                if self._turn_to(GRASP_YAW, ALIGN_TURN_MAX, ALIGN_TURN_GAIN):
                    if self.align_settle_t0 == 0.0:
                        self.align_settle_t0 = self._now()
                    elif self._now() - self.align_settle_t0 > 0.3:
                        self.target_tracker.clear()
                        self.target_locked = False
                        self._enter(DEPLOY)
                else:
                    self.align_settle_t0 = 0.0
        elif self.phase == DEPLOY:
            a.stop_base()
            level = self.target.slot[1] if self.target is not None else "L2"
            grasp_slide = SLIDE_GRASP_BY_LEVEL.get(level, SLIDE_GRASP)
            base_pitch = (SCAN_PITCH_BY_LEVEL.get(level, HEAD_PITCH)
                          + DEPLOY_PITCH_K * (grasp_slide - SCAN_SLIDE))
            dyaw, dpitch = DEPLOY_LOOK[min(self.deploy_pitch_idx, len(DEPLOY_LOOK) - 1)]
            yaw = DEPLOY_HEAD_YAW + dyaw
            pitch = base_pitch + dpitch
            a.set_head(yaw, pitch)
            a.set_slide(self.grasp_slide)
            a.set_gripper("right", GRIP_OPEN)
            now = self._now()
            if not self.target_locked and a.head_settled(yaw, pitch):
                if now - self.deploy_pitch_t > DETECT_DWELL:
                    if self._lock_from_products():
                        if a.arm_to("right", self.deploy_world, GRASP_ROT):
                            self.target_locked = True
                            fp = a.world_to_footprint(self.deploy_world)
                            self.get_logger().info(
                                f"locked {self.target_kind} world={np.round(self.deploy_world, 3)}")
                            self.get_logger().info(
                                f"[deploy] world={np.round(self.deploy_world, 3)} "
                                f"fp={np.round(fp, 3)} grasp={np.round(self.grasp_world, 3)} "
                                f"cmd={np.round(a.tc[12:18], 3)} meas={np.round(a.arm_meas('right'), 3)}")
                            self.motion_settle_t0 = 0.0
                            self._enter(WAIT_ARM)
                        else:
                            self.get_logger().warn(
                                f"IK failed for {np.round(self.deploy_world, 3)}, retrying")
                            self.target_tracker.clear()
                    elif (self.deploy_pitch_idx < len(DEPLOY_LOOK) - 1
                          and now - self.deploy_pitch_t > DETECT_DWELL + DEPLOY_PITCH_DWELL):
                        # target not seen here: sweep the head and retry
                        self.deploy_pitch_idx += 1
                        self.deploy_pitch_t = now
                        self.target_tracker.clear()
                        ny, np_ = DEPLOY_LOOK[self.deploy_pitch_idx]
                        self.get_logger().info(
                            f"[deploy] {level} target not seen at yaw={yaw:.2f} "
                            f"pitch={pitch:.2f}; trying yaw={DEPLOY_HEAD_YAW + ny:.2f} "
                            f"pitch={base_pitch + np_:.2f}")
        elif self.phase == WAIT_ARM:
            a.stop_base()
            if self._now() - self.last_arm_dbg > 0.5:
                self.last_arm_dbg = self._now()
                per = np.abs(a.tc[12:18] - a.arm_meas("right"))
                pos_err, max_vel = a.arm_errors("right")
                ee = a.ee_world("right")
                self.get_logger().info(
                    f"[arm] pos_err={pos_err:.3f} max_vel={max_vel:.3f} "
                    f"per={np.round(per, 3)} jidx={int(np.argmax(per)) + 1} "
                    f"ee={np.round(ee, 3)} "
                    f"d_deploy={np.linalg.norm(ee - self.deploy_world):.3f} "
                    f"d_grasp={np.linalg.norm(ee - self.grasp_world):.3f}")
            # Accept the pose when the end-effector is at the deploy target even
            # if a wrist joint has a residual (L3 reaches a joint-5 limit and
            # stops ~0.12 rad short while the EE is already within ~3 cm).
            ee_ok = float(np.linalg.norm(
                a.ee_world("right") - self.deploy_world)) < 0.05
            if ee_ok or a.arm_settled("right", pos_tol=0.15, vel_tol=0.08):
                if self._fine_adjust_target():
                    # arm re-solved to the marker; wait for it to settle again
                    self.motion_settle_t0 = 0.0
                    return
                if self.motion_settle_t0 == 0.0:
                    self.motion_settle_t0 = self._now()
                elif self._now() - self.motion_settle_t0 >= 0.3:
                    self.motion_settle_t0 = 0.0
                    self._enter(CREEP)
            else:
                self.motion_settle_t0 = 0.0
        elif self.phase == CREEP:
            ee = a.ee_world("right")
            axis = np.array([math.cos(GRASP_YAW), math.sin(GRASP_YAW)])
            remaining = float(np.dot(self.grasp_world[:2] - ee[:2], axis))
            # drive a little PAST the product centre so the fingers wrap it
            speed = creep_speed(remaining, stop_gap=-CREEP_STOP_GAP)
            if self._now() - self.last_arm_dbg > 0.3:
                self.last_arm_dbg = self._now()
                self.get_logger().info(
                    f"[creep] ee={np.round(ee, 3)} grasp={np.round(self.grasp_world, 3)} "
                    f"remaining={remaining:.3f} speed={speed:.3f} "
                    f"base=({a.base_xy[0]:.3f},{a.base_xy[1]:.3f})")
            if speed > 0.0:
                a.set_base_velocity(speed, 1.5 * wrap_to_pi(GRASP_YAW - a.base_yaw))
            else:
                a.stop_base()
                self.motion_settle_t0 = 0.0
                self._enter(BRAKE)
        elif self.phase == BRAKE:
            a.stop_base()
            if a.base_stopped():
                if self.motion_settle_t0 == 0.0:
                    self.motion_settle_t0 = self._now()
                elif self._now() - self.motion_settle_t0 >= 0.3:
                    self.motion_settle_t0 = 0.0
                    self._enter(CLOSE)
            else:
                self.motion_settle_t0 = 0.0
        elif self.phase == CLOSE:
            a.stop_base()
            a.set_gripper("right", GRIP_CLOSE)
            meas = a.gripper_meas("right")
            # A stalled width well above the closed target means an object is
            # between the fingers (an empty close reaches GRIP_CLOSE).
            held = 0.15 < meas < 0.9
            at_target = abs(meas - GRIP_CLOSE) < 0.05
            elapsed = self._now() - self.state_t0
            if self._now() - self.last_arm_dbg > 0.3:
                self.last_arm_dbg = self._now()
                self.get_logger().info(
                    f"[close] meas={meas:.3f} vel={a.gripper_vel('right'):.3f} "
                    f"target={GRIP_CLOSE:.3f} at_target={at_target} "
                    f"held={held} t={elapsed:.2f}")
            if (at_target and elapsed > 0.3) or elapsed > 1.5:
                self.get_logger().info(
                    f"[close] done meas={meas:.3f} held={held} t={elapsed:.2f}")
                self._enter(LIFT)
        elif self.phase == LIFT:
            a.stop_base()
            a.set_slide(self.grasp_slide - LIFT_AMOUNT)
            if abs(a.slide_meas - (self.grasp_slide - LIFT_AMOUNT)) < 0.02:
                self._enter(RETREAT)
        elif self.phase == RETREAT:
            # Back straight out of the shelf (no heading correction) so the
            # held item is not swung; the turn happens afterwards in RETURN.
            if a.base_xy[1] > YELLOW_MID_Y + 0.06:
                a.set_base_velocity(-RETREAT_SPEED, 0.0)
            else:
                a.stop_base()
                self.prim_start_xy = None
                self._enter(RETURN)
        elif self.phase == RETURN:
            # hard-coded axis-aligned run to the obstacle-zone entry, then plan
            if self._now() - self.last_arm_dbg > 1.0:
                self.last_arm_dbg = self._now()
                self.get_logger().info(
                    f"[return] base=({a.base_xy[0]:.2f},{a.base_xy[1]:.2f}) "
                    f"d_entry="
                    f"{math.hypot(a.base_xy[0] - OBSTACLE_ENTRY[0], a.base_xy[1] - OBSTACLE_ENTRY[1]):.2f}")
            if self._drive_to(OBSTACLE_ENTRY[0], OBSTACLE_ENTRY[1]):
                self.path = None
                self.path_goal = None
                self._enter(NAV_TABLE)
        elif self.phase == NAV_TABLE:
            if self._navigate(TABLE_APPROACH, DELIVERY_YAW):
                self._enter(PLACE)
        elif self.phase == PLACE:
            # Reach forward over the fixed table, lower to just above the top,
            # release, then lift clear and home.  Keeps the item upright so it
            # lands stably instead of dropping from height.
            a.stop_base()
            place_xy = TABLE_PLACE_XY
            if self.place_sub == 0:
                # 1) lower the spine (arm joints unchanged) as in the baseline
                a.set_slide(PLACE_LOWER_SLIDE)
                if abs(a.slide_meas - PLACE_LOWER_SLIDE) < 0.03:
                    self.place_sub = 1
            elif self.place_sub == 1:
                # 2) reach forward to hover above the table
                hover = np.array([place_xy[0], place_xy[1], PLACE_HOVER_Z])
                if a.arm_to("right", hover, GRASP_ROT):
                    self.place_sub = 2
                    self.place_sub_t0 = self._now()
                else:
                    self.get_logger().warn(
                        "place hover IK failed; releasing in place")
                    self.place_sub = 3
                    self.place_sub_t0 = self._now()
            elif self.place_sub == 2:
                # 3) wait for the hover pose, then lower onto the table.  A
                # wrist joint may hold a residual (L3 reaches a limit), so
                # accept the EE reaching the target, with a time fallback.
                hover = np.array([place_xy[0], place_xy[1], PLACE_HOVER_Z])
                at_hover = float(np.linalg.norm(
                    a.ee_world("right") - hover)) < 0.05
                if (at_hover or a.arm_settled("right", pos_tol=0.15, vel_tol=0.08)
                        or self._now() - self.place_sub_t0 > 2.0):
                    drop = np.array([place_xy[0], place_xy[1], PLACE_DROP_Z])
                    if a.arm_to("right", drop, GRASP_ROT):
                        self.place_sub = 3
                        self.place_sub_t0 = self._now()
            elif self.place_sub == 3:
                # 4) once at the set-down pose, open the gripper (EE-based, with
                # a time fallback so the item is always released)
                drop = np.array([place_xy[0], place_xy[1], PLACE_DROP_Z])
                at_drop = float(np.linalg.norm(
                    a.ee_world("right") - drop)) < 0.05
                if (at_drop or a.arm_settled("right", pos_tol=0.15, vel_tol=0.08)
                        or self._now() - self.place_sub_t0 > 1.5):
                    a.set_gripper("right", GRIP_OPEN)
                    self.place_sub = 4
                    self.state_t0 = self._now()
                    self.place_sub_t0 = self._now()
            elif self.place_sub == 4:
                # 5) let the item settle on the table, then lift clear
                if self._now() - self.state_t0 > PLACE_RELEASE_DWELL:
                    raise_pose = np.array([place_xy[0], place_xy[1], PLACE_RAISE_Z])
                    if a.arm_to("right", raise_pose, GRASP_ROT):
                        self.place_sub = 5
                        self.place_sub_t0 = self._now()
                    else:
                        a.home()
                        self._enter(NEXT)
            elif self.place_sub == 5:
                # 6) tuck the arm, then continue
                if a.arm_settled("right", pos_tol=0.10, vel_tol=0.08):
                    a.home()
                    self._enter(NEXT)
            if self._now() - self.last_arm_dbg > 0.5:
                self.last_arm_dbg = self._now()
                ee = a.ee_world("right")
                drop = np.array([place_xy[0], place_xy[1], PLACE_DROP_Z])
                self.get_logger().info(
                    f"[place] sub={self.place_sub} slide={a.slide_meas:.3f} "
                    f"ee={np.round(ee, 3)} d_drop="
                    f"{float(np.linalg.norm(ee - drop)):.3f} "
                    f"grip={a.gripper_meas('right'):.3f}")
        elif self.phase == NEXT:
            if self.target is not None:
                self.inventory.consume(self.target.slot)
            if self.target_order in self.pending:
                self.pending.remove(self.target_order)
            self.target = None
            self.target_order = None
            self.target_kind = None
            self.target_locked = False
            self.target_tracker.clear()
            if not self.pending:
                self.stow_next = DONE
            elif self._select_target():
                self.stow_next = NAV_RETURN
            else:
                self.explore_plan = None
                self.explore_i = 0
                self.stow_next = NAV_RETURN
            self._enter(STOW)
        elif self.phase == NAV_RETURN:
            if self._navigate(OBSTACLE_ENTRY, final_yaw=None, tol=0.15):
                self.path = None
                self.path_goal = None
                if self.target is not None:
                    self._start_nav_shelf()
                else:
                    self.explore_plan = None
                    self.explore_i = 0
                    self._enter(SCAN)
        elif self.phase == DONE:
            a.stop_base()
        else:
            a.stop_base()

        a.step()
        self._tick_dt = self._now() - getattr(self, "_tick_t0", self._now())
        self._log()

    def _build_scan_plan(self):
        """Visit only shelves whose full camera sweep has not completed."""
        shelves = [s for s in SHELF_X if s not in self.scanned_shelves]
        if self.adapter.base_xy is not None:
            bx = float(self.adapter.base_xy[0])
            ordered = sorted(shelves, key=lambda s: SHELF_X[s])
            nearest = min(range(len(ordered)), key=lambda i: abs(SHELF_X[ordered[i]] - bx))
            left = list(reversed(ordered[:nearest]))
            right = ordered[nearest + 1:]
            if left and right and abs(SHELF_X[right[0]] - bx) < abs(SHELF_X[left[0]] - bx):
                shelves = [ordered[nearest], *right, *left]
            else:
                shelves = [ordered[nearest], *left, *right]
        plan = []
        for shelf in shelves:
            plan += [("goto", (SHELF_X[shelf], SCAN_Y)),
                     ("turn", math.pi / 2.0), ("scan", shelf)]
        return plan

    def _prim_reset(self):
        self.prim_start_xy = None
        self.view_t0 = 0.0
        self.view_start = 0.0
        self.view_settle_t0 = 0.0
        self.view_products_seq = -1

    def _drive_to(self, x, y, speed=EXPLORE_SPEED, turn_max=EXPLORE_TURN_MAX):
        """Drive to a world point; turn in place first when badly misaligned.

        This is a hard-coded primitive for the departure / shelf zone: it never
        consults the LaserScan, the planner or any recovery (competition rule).
        """
        a = self.adapter
        dx = x - float(a.base_xy[0])
        dy = y - float(a.base_xy[1])
        dist = math.hypot(dx, dy)
        if dist < EXPLORE_DRIVE_TOL:
            a.stop_base()
            return True
        err = wrap_to_pi(math.atan2(dy, dx) - a.base_yaw)
        v = 0.0 if abs(err) > 0.4 else min(speed, max(0.03, 0.6 * dist))
        a.set_base_velocity(v, max(-turn_max, min(turn_max, 1.0 * err)))
        return False

    def _turn_to(self, yaw, max_rate=EXPLORE_TURN_MAX, gain=0.5):
        a = self.adapter
        err = wrap_to_pi(yaw - a.base_yaw)
        if abs(err) < EXPLORE_TURN_TOL:
            a.stop_base()
            return True
        a.set_base_velocity(0.0, max(-max_rate, min(max_rate, gain * err)))
        return False

    def _drive_dist(self, dist):
        a = self.adapter
        if self.prim_start_xy is None:
            self.prim_start_xy = a.base_xy.copy()
            self.prim_start_yaw = a.base_yaw
        dx = float(a.base_xy[0] - self.prim_start_xy[0])
        dy = float(a.base_xy[1] - self.prim_start_xy[1])
        traveled = math.cos(self.prim_start_yaw) * dx + math.sin(self.prim_start_yaw) * dy
        if abs(traveled - dist) < EXPLORE_DRIVE_TOL:
            a.stop_base()
            return True
        remain = dist - traveled
        v = max(-EXPLORE_SPEED, min(EXPLORE_SPEED, 0.6 * remain))
        yaw_err = wrap_to_pi(self.prim_start_yaw - a.base_yaw)
        a.set_base_velocity(v, max(-EXPLORE_TURN_MAX, min(EXPLORE_TURN_MAX, 1.0 * yaw_err)))
        return False

    def _do_scan(self):
        a = self.adapter
        a.stop_base()
        a.set_slide(SCAN_SLIDE)
        yaw = SCAN_YAWS[self.scan_yaw_idx]
        pitch = SCAN_PITCHES[self.scan_pitch_idx]
        a.set_head(yaw, pitch)
        now = self._now()
        if not a.head_settled(yaw, pitch):
            self.view_settle_t0 = 0.0
            self.view_t0 = 0.0
            self.view_start = 0.0
            return False
        if self.view_settle_t0 == 0.0:
            self.view_settle_t0 = now
            self.view_products_seq = self.products_seq
            return False
        if now - self.view_settle_t0 < SCAN_SETTLE:
            return False
        if self.products_seq <= self.view_products_seq:
            return False
        if self.view_t0 == 0.0:
            self.view_t0 = now
            self.view_start = now
            self.view_inv = len(self.inventory.all())
        inv_n = len(self.inventory.all())
        if inv_n > self.view_inv:
            self.view_inv = inv_n
            self.view_t0 = now
        if (now - self.view_t0 < SCAN_DWELL
                and now - self.view_start < SCAN_DWELL_MAX):
            return False
        # persist this look's detections (kind -> slot candidates) immediately
        self.inventory.update(self.products)
        self.scan_pitch_idx += 1
        self.view_t0 = 0.0
        self.view_start = 0.0
        self.view_settle_t0 = 0.0
        self.view_products_seq = -1
        self.scan_views += 1
        if self.scan_pitch_idx >= len(SCAN_PITCHES):
            self.scan_pitch_idx = 0
            self.scan_yaw_idx += 1
            if self.scan_yaw_idx >= len(SCAN_YAWS):
                self.scan_yaw_idx = 0
                return True
        return False

    def _tick_scan(self):
        a = self.adapter
        # Each shelf is swept completely (all levels) the first time it is
        # visited, so its inventory is fully known before deciding what to do;
        # only then is any known target grasped (preferring the current shelf).
        if a.base_xy is None:
            a.stop_base()
            return
        if self.explore_plan is None:
            self.explore_plan = self._build_scan_plan()
            self.explore_i = 0
            self.scan_yaw_idx = 0
            self.scan_pitch_idx = 0
            self.get_logger().info(f"exploration plan: {self.explore_plan}")
        if self.explore_i >= len(self.explore_plan):
            if self._select_target():
                self._start_nav_shelf()
                return
            self.scan_complete = len(self.scanned_shelves) == len(SHELF_X)
            self.get_logger().info(
                f"scan coverage={sorted(self.scanned_shelves)} views={self.scan_views}")
            self._enter(ERROR if self.pending else DONE)
            return
        step = self.explore_plan[self.explore_i]
        kind = step[0]
        arg = step[1] if len(step) > 1 else None
        if kind == "goto":
            done = self._drive_to(arg[0], arg[1], SCAN_SPEED, SCAN_TURN_MAX)
        elif kind == "turn":
            done = self._turn_to(arg, SCAN_TURN_MAX)
        elif kind == "drive":
            done = self._drive_dist(arg)
        else:
            done = self._do_scan()
        if done:
            if kind == "scan" and arg is not None:
                self.scanned_shelves.add(arg)
                self.get_logger().info(
                    f"scanned shelf {arg}; coverage={sorted(self.scanned_shelves)}")
                # shelf inventory is complete: grasp any known target (the
                # current shelf is preferred), otherwise continue scanning.
                if self._select_target():
                    self._start_nav_shelf()
                    return
            self.explore_i += 1
            self._prim_reset()

    def _start_nav_shelf(self):
        lane = self._approach_lane()
        self.path = None
        self.path_goal = None
        self.prim_start_xy = None
        self.align_stage = "pos"
        self.align_settle_t0 = 0.0
        self.fine_adjusted = False
        level = self.target.slot[1] if self.target is not None else "L2"
        self.grasp_slide = SLIDE_GRASP_BY_LEVEL.get(level, SLIDE_GRASP)
        self.get_logger().info(
            f"target kind={self.target_kind} slot={self.target.slot} lane={lane} "
            f"slide={self.grasp_slide}")
        self._enter(ALIGN)

    def _on_timeout(self):
        phase = self.phase
        self.get_logger().warn(
            f"timeout in {PHASE_NAME[phase]} "
            f"target={self.target.slot if self.target else None}")
        self.adapter.stop_base()
        if phase in (ALIGN, DEPLOY, WAIT_ARM, CREEP, BRAKE, CLOSE):
            self.adapter.home()
            if self.target is not None:
                self.inventory.release(self.target.slot)
                self.failed_slots.add(self.target.slot)
            self.target = None
            self.target_order = None
            self.target_kind = None
            self.target_locked = False
            self.target_tracker.clear()
            self._recover()
        elif phase == NAV_RETURN:
            if self.target is not None:
                self.inventory.release(self.target.slot)
            self.target = None
            self.target_order = None
            self.target_kind = None
            self._enter(ERROR)
        elif phase == STOW:
            self._enter(ERROR)
        else:
            # holding or placing: do NOT home (avoid dropping); stop and finish
            self._enter(DONE)

    def _recover(self):
        if self.pending and self._select_target():
            # keep the inventory and go straight to the next candidate: a
            # failed grasp must not force a full re-scan.
            self.stow_next = ALIGN
            self._enter(STOW)
        elif self.pending and len(self.scanned_shelves) < len(SHELF_X):
            # continue with the shelves not covered yet
            self.explore_plan = None
            self.explore_i = 0
            self.stow_next = SCAN
            self._enter(STOW)
        elif self.pending and self.recover_scans < 1:
            # every shelf was swept but no candidate for a pending kind: sweep
            # once more (a target may have been missed) before giving up.
            self.recover_scans += 1
            self.scanned_shelves.clear()
            self.explore_plan = None
            self.explore_i = 0
            self.stow_next = SCAN
            self._enter(STOW)
        elif self.pending:
            self._enter(ERROR)
        else:
            self._enter(DONE)

    def _log(self):
        if self._now() - self.last_log < 1.0:
            return
        self.last_log = self._now()
        a = self.adapter
        inv = self.inventory.summary()
        pending = [f"{t.id}:{t.kind}" for t in self.pending]
        self.get_logger().info(
            f"phase={PHASE_NAME[self.phase]} base=({a.base_xy[0]:.2f},{a.base_xy[1]:.2f}) "
            f"yaw={a.base_yaw:.2f} cmd=({a.des_lin:.2f},{a.des_ang:.2f}) "
            f"meas_v=({a.base_lin_meas:.3f},{a.base_ang_meas:.3f}) "
            f"tick={getattr(self, '_tick_dt', 0.0):.2f} "
            f"front_clear={self._front_clear()} nav={self.nav_idx}/{len(self.route)}:{self.nav_mode} "
            f"lock={self.target_tracker.sample_count} pending={pending} inv={inv}")


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
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
