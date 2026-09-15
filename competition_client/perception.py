#!/usr/bin/env python3
"""Unified perception node: 9-class product detection + ArUco, in world frame.

Subscribes to the official head camera + robot state, runs the product YOLO and
an ArUco detector on the same RGB frame, deprojects through the head camera and
reports both products and shelf markers in the WORLD frame.

Products: /competition/product_detections  (custom JSON string)
ArUco   : /competition/aruco_detections    (custom JSON string)
Debug   : /competition/result_image        (annotated RGB)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String

try:
    from competition_client.detector import ProductDetector, DEFAULT_WEIGHTS
except Exception:  # allow running from inside the package dir
    from detector import ProductDetector, DEFAULT_WEIGHTS

from kinematics.mmk2_kdl import MMK2Kdl

ARUCO_DICT = cv2.aruco.DICT_4X4_50
MARKER_SIZE_M = 0.03
VALID_ARUCO_IDS = set(range(45))
ASSOCIATION_MAX_DIST_M = 0.22
ASSOCIATION_MIN_MARGIN_M = 0.03

# Right hand-eye camera (rgt_handeye) pose relative to the right-arm
# end-effector frame (MMK2Kdl right EE / mjcf rgt_endpoint).  Calibrated in
# MuJoCo; the optical frame is the MuJoCo camera frame rotated by pi about X,
# the same convention used by the head camera's ``headeye`` site.
T_EE_CAMOPT = np.array([
    [0.0, -0.5, 0.8660254, -0.1299],
    [-1.0, 0.0, 0.0, -0.00482],
    [0.0, -0.8660254, -0.5, 0.10665],
    [0.0, 0.0, 0.0, 1.0],
])

# Only trust detections in the near frontal region of the base.  This drops far
# items that would otherwise be mis-associated to the wrong shelf.
NEAR_FWD_MIN, NEAR_FWD_MAX = 0.15, 1.40
NEAR_LAT_MAX = 0.35
NEAR_Z_MIN, NEAR_Z_MAX = 0.30, 1.50

# Fixed shelf geometry (world frame).  This is public scene structure, NOT the
# randomised product-to-slot assignment.
SHELF_X = {"A": -1.735, "B": -0.850, "C": 0.035, "D": 0.920, "E": 1.805}
COLUMN_DX = {"C1": -0.220, "C2": 0.0, "C3": 0.220}
SHELF_Y = 3.323
LEVEL_Z = {"L1": 0.499, "L2": 0.851, "L3": 1.189}

# ArUco ID -> slot.  The mapping is fixed by the scene: A:0-8, B:9-17, ... and
# within a shelf L1->L3, C1->C3.  This is the "货位-ArUco" relation, which is
# fixed (only the product<->slot relation is randomised).
SHELF_ORDER = ["A", "B", "C", "D", "E"]
LEVEL_ORDER = ["L1", "L2", "L3"]
COLUMN_ORDER = ["C1", "C2", "C3"]


def aruco_id_to_slot(aruco_id: int):
    if 0 <= aruco_id < 45:
        rem = aruco_id % 9
        return (SHELF_ORDER[aruco_id // 9], LEVEL_ORDER[rem // 3], COLUMN_ORDER[rem % 3])
    return None


def slot_from_world(p) -> Optional[Dict]:
    """Map a world point to the nearest (shelf, level, column) slot."""
    shelf = min(SHELF_X, key=lambda s: abs(p[0] - (SHELF_X[s] + COLUMN_DX["C2"])))
    # column is measured within the shelf; choose the closest of the 3 columns
    col = min(COLUMN_DX, key=lambda c: abs(p[0] - (SHELF_X[shelf] + COLUMN_DX[c])))
    level = min(LEVEL_Z, key=lambda l: abs(p[2] - LEVEL_Z[l]))
    # sanity gate: within ~0.18 m horizontally and 0.25 m vertically of the slot
    if abs(p[1] - SHELF_Y) > 0.35:
        return None
    if abs(p[0] - (SHELF_X[shelf] + COLUMN_DX[col])) > 0.18:
        return None
    if abs(p[2] - LEVEL_Z[level]) > 0.28:
        return None
    return {"shelf": shelf, "level": level, "column": col}


def validated_aruco_anchors(arucos):
    """Return markers whose configured slot agrees with world quantization."""
    anchors = []
    for marker in arucos:
        slot = aruco_id_to_slot(marker["id"])
        measured = slot_from_world(marker["world"])
        measured_slot = ((measured["shelf"], measured["level"], measured["column"])
                         if measured is not None else None)
        if slot is not None and measured_slot == slot:
            anchors.append((slot, np.asarray(marker["world"], dtype=float), marker["id"]))
    return anchors


class PerceptionNode(Node):
    def __init__(self, detector=None, weights=DEFAULT_WEIGHTS,
                 confidence: float = 0.35, device: str = "auto",
                 enable_aruco: bool = True):
        super().__init__("competition_perception")
        from kinematics.mmk2_fk import MMK2FK

        self.bridge = CvBridge()
        self.fk = MMK2FK()
        self.enable_aruco = enable_aruco

        self.K: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None
        self._depth_msg: Optional[Image] = None
        self.base_pos = None
        self.base_quat = None
        self.slide = 0.0
        self.head = [0.0, 0.0]
        self.rarm = [0.0] * 6
        self.K_he: Optional[np.ndarray] = None
        self.D_he: Optional[np.ndarray] = None
        self.kdl = MMK2Kdl()

        self.detector = detector or ProductDetector(weights, confidence, device)

        self._detector_aruco = None
        self._aruco_params = None
        if enable_aruco:
            self._detector_aruco = cv2.aruco.ArucoDetector(
                cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
                cv2.aruco.DetectorParameters())

        self.latest: Dict = {"products": [], "aruco": [], "stamp": 0.0}

        self.create_subscription(CameraInfo, "/head_camera/color/camera_info",
                                 self._info_cb, 10)
        self.create_subscription(Image, "/head_camera/aligned_depth_to_color/image_raw",
                                 self._depth_cb, 10)
        self.create_subscription(Image, "/head_camera/color/image_raw",
                                 self._rgb_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)
        self.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom",
                                 self._odom_cb, 10)
        self.create_subscription(CameraInfo, "/right_camera/color/camera_info",
                                 self._handeye_info_cb, 10)
        self.create_subscription(Image, "/right_camera/color/image_raw",
                                 self._handeye_rgb_cb, 10)

        self.products_pub = self.create_publisher(String, "/competition/product_detections", 10)
        self.aruco_pub = self.create_publisher(String, "/competition/aruco_detections", 10)
        self.handeye_aruco_pub = self.create_publisher(
            String, "/competition/handeye_aruco_detections", 10)
        self.debug_pub = self.create_publisher(Image, "/competition/result_image", 5)
        self.get_logger().info("perception up (9-class + ArUco + hand-eye)")

    # ---- callbacks ----
    def _info_cb(self, msg: CameraInfo):
        self.K = np.asarray(msg.k, dtype=float).reshape(3, 3)
        self.D = np.asarray(msg.d, dtype=float)

    def _depth_cb(self, msg: Image):
        self._depth_msg = msg

    def _js_cb(self, msg: JointState):
        jp = {n: msg.position[i] for i, n in enumerate(msg.name) if i < len(msg.position)}
        self.slide = jp.get("slide_joint", self.slide)
        self.head = [jp.get("head_yaw_joint", self.head[0]),
                     jp.get("head_pitch_joint", self.head[1])]
        self.rarm = [jp.get(f"right_arm_joint{i + 1}", self.rarm[i])
                     for i in range(6)]

    def _handeye_info_cb(self, msg: CameraInfo):
        self.K_he = np.asarray(msg.k, dtype=float).reshape(3, 3)
        self.D_he = np.asarray(msg.d, dtype=float)

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_pos = [p.x, p.y, p.z]
        self.base_quat = [q.w, q.x, q.y, q.z]

    # ---- transforms ----
    def camera_world_tmat(self):
        if self.base_pos is None or self.base_quat is None:
            return None
        self.fk.set_base_pose(self.base_pos, self.base_quat)
        self.fk.set_slide_joint(float(self.slide))
        self.fk.set_head_joints([float(self.head[0]), float(self.head[1])])
        pos, quat = self.fk.get_head_camera_pose()
        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
        return T

    def handeye_world_tmat(self):
        """World pose of the right hand-eye optical frame from arm FK + calib."""
        if self.base_pos is None or self.base_quat is None:
            return None
        q = np.array([float(self.slide)] + [float(v) for v in self.rarm])
        try:
            _, T_ee = self.kdl.forward_kinematics(q, index="right")
        except Exception:  # noqa: BLE001
            return None
        T_base_cam = np.asarray(T_ee, dtype=float) @ T_EE_CAMOPT
        T_wb = np.eye(4)
        T_wb[:3, :3] = Rotation.from_quat(
            np.asarray(self.base_quat)[[1, 2, 3, 0]]).as_matrix()
        T_wb[:3, 3] = np.asarray(self.base_pos, dtype=float)
        return T_wb @ T_base_cam

    def pixel_to_cam(self, u, v, depth_m):
        pixel = np.array([[[float(u), float(v)]]], dtype=np.float64)
        distortion = self.D if self.D is not None and self.D.size else None
        normalized = cv2.undistortPoints(pixel, self.K, distortion).reshape(2)
        return np.array([normalized[0] * depth_m, normalized[1] * depth_m, depth_m])

    @staticmethod
    def patch_depth_m(depth_img, u, v, r=4):
        h, w = depth_img.shape[:2]
        y0, y1 = max(0, v - r), min(h, v + r + 1)
        x0, x1 = max(0, u - r), min(w, u + r + 1)
        patch = depth_img[y0:y1, x0:x1].astype(np.float32)
        valid = patch[patch > 0]
        return float(np.median(valid)) * 1e-3 if len(valid) else 0.0

    # ---- main ----
    def _rgb_cb(self, msg: Image):
        if self.K is None or self._depth_msg is None:
            return
        frame_stamp = float(msg.header.stamp.sec) + msg.header.stamp.nanosec * 1e-9
        depth_stamp = (float(self._depth_msg.header.stamp.sec)
                       + self._depth_msg.header.stamp.nanosec * 1e-9)
        if abs(frame_stamp - depth_stamp) > 0.05:
            return
        T_cw = self.camera_world_tmat()
        if T_cw is None:
            return

        rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        depth = self.bridge.imgmsg_to_cv2(self._depth_msg)

        arucos = self._detect_aruco(rgb, T_cw) if self.enable_aruco else []
        arucos = [a for a in arucos if self._in_near_region(a["world"])]
        anchors = validated_aruco_anchors(arucos)

        products = []
        used_marker_ids = set()
        for d in self.detector.detect(rgb):
            depth_m = self.patch_depth_m(depth, d["x"], d["y"])
            if depth_m <= 0.0:
                continue
            p_cam = self.pixel_to_cam(d["x"], d["y"], depth_m)
            p_world = (T_cw @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]
            if not self._in_near_region(p_world):
                continue
            slot, aruco_id, source = self._associate(
                p_world, anchors, used_marker_ids=used_marker_ids)
            if aruco_id is not None:
                used_marker_ids.add(aruco_id)
            products.append({
                "kind": d["kind"], "conf": float(d["conf"]),
                "world": [float(x) for x in p_world],
                "stamp": frame_stamp,
                "slot": ({"shelf": slot[0], "level": slot[1], "column": slot[2]}
                         if slot else None),
                "aruco_id": aruco_id,
                "slot_source": source,
            })

        self.latest = {
            "products": products,
            "aruco": arucos,
            "stamp": frame_stamp,
        }
        self.products_pub.publish(String(data=json.dumps(products)))
        self.aruco_pub.publish(String(data=json.dumps(arucos)))

    def _handeye_rgb_cb(self, msg: Image):
        if self.K_he is None or self.base_pos is None or self.base_quat is None:
            return
        T_wc = self.handeye_world_tmat()
        if T_wc is None:
            return
        rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        arucos = self._detect_aruco(rgb, T_wc, self.K_he, self.D_he)
        if not arucos:
            return
        stamp = float(msg.header.stamp.sec) + msg.header.stamp.nanosec * 1e-9
        payload = [{"id": a["id"], "world": a["world"],
                    "cam_xyz": a["cam_xyz"], "stamp": stamp} for a in arucos]
        self.handeye_aruco_pub.publish(String(data=json.dumps(payload)))

    def _in_near_region(self, p_world) -> bool:
        """True if a world point is in the base's near frontal region."""
        if self.base_pos is None or self.base_quat is None:
            return True
        yaw = Rotation.from_quat(
            np.asarray(self.base_quat)[[1, 2, 3, 0]]).as_euler("xyz")[2]
        dx = float(p_world[0]) - float(self.base_pos[0])
        dy = float(p_world[1]) - float(self.base_pos[1])
        c, s = math.cos(-yaw), math.sin(-yaw)
        fwd = c * dx - s * dy
        lat = s * dx + c * dy
        z = float(p_world[2])
        return (NEAR_FWD_MIN <= fwd <= NEAR_FWD_MAX
                and abs(lat) <= NEAR_LAT_MAX
                and NEAR_Z_MIN <= z <= NEAR_Z_MAX)

    @staticmethod
    def _associate(p_world, anchors, used_marker_ids=None,
                   max_dist: float = ASSOCIATION_MAX_DIST_M,
                   min_margin: float = ASSOCIATION_MIN_MARGIN_M):
        """Associate only an unambiguous, unused marker; otherwise use geometry."""
        p = np.asarray(p_world, dtype=float)
        candidates = []
        for slot, aw, aid in anchors:
            dist = float(np.linalg.norm(p - aw))
            candidates.append((dist, slot, aid))
        candidates.sort(key=lambda item: item[0])

        best = candidates[0] if candidates else None
        ambiguous = (len(candidates) > 1
                     and candidates[1][0] - candidates[0][0] < min_margin)
        used_marker_ids = used_marker_ids or set()
        if (best is not None and best[0] < max_dist and not ambiguous
                and best[2] not in used_marker_ids):
            return best[1], best[2], "aruco"
        s = slot_from_world(p_world)
        if s is not None:
            return (s["shelf"], s["level"], s["column"]), None, "geometry"
        return None, None, None

    def _detect_aruco(self, rgb, T_cw, K=None, D=None) -> List[Dict]:
        K = self.K if K is None else K
        D = self.D if D is None else D
        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self._detector_aruco.detectMarkers(gray)
        out = []
        if ids is None:
            return out
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in VALID_ARUCO_IDS:
                continue
            half = MARKER_SIZE_M * 0.5
            obj = np.array([[-half, half, 0], [half, half, 0],
                            [half, -half, 0], [-half, -half, 0]], dtype=np.float32)
            distortion = D if D is not None and D.size else None
            ok, rvec, tvec = cv2.solvePnP(
                obj, np.asarray(marker_corners, dtype=np.float32).reshape(4, 2),
                K, distortion, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if not ok:
                continue
            tvec = tvec.reshape(3)
            p_world = (T_cw @ np.array([tvec[0], tvec[1], tvec[2], 1.0]))[:3]
            out.append({
                "id": marker_id,
                "world": [float(x) for x in p_world],
                "cam_xyz": [float(x) for x in tvec],
            })
        return out

    def snapshot(self) -> Dict:
        return self.latest


def main():
    import rclpy

    rclpy.init()
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
