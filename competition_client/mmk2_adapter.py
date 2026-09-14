#!/usr/bin/env python3
"""MMK2 control adapter for the official supermarket sorting server.

Wraps the 19-D target_control vector used by the reference baseline and
publishes it to the official position-controller topics.  All joint targets are
slew-rate limited so a freshly computed IK target never snaps (the cause of the
"瞬移" teleport).  Base velocity is acceleration limited.

Layout of ``tc`` (19 elements):
    [0]  base linear.x        [1]  base angular.z
    [2]  slide_joint          [3]  head_yaw        [4]  head_pitch
    [5:11]  left arm joint1..6
    [11]    left gripper
    [12:18] right arm joint1..6
    [18]    right gripper
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

from common.control import step_func
from kinematics.mmk2_kdl import MMK2Kdl

# JointState names (order documented by the server).
JOINT_NAMES = [
    "slide_joint", "head_yaw_joint", "head_pitch_joint",
    "left_arm_joint1", "left_arm_joint2", "left_arm_joint3",
    "left_arm_joint4", "left_arm_joint5", "left_arm_joint6",
    "left_arm_eef_gripper_joint",
    "right_arm_joint1", "right_arm_joint2", "right_arm_joint3",
    "right_arm_joint4", "right_arm_joint5", "right_arm_joint6",
    "right_arm_eef_gripper_joint",
]

INIT_ARM_L = [0.0, -0.166, 0.032, 0.0, 1.571, 2.223]
INIT_ARM_R = [0.0, -0.166, 0.032, 0.0, -1.571, -2.223]

GRIP_OPEN, GRIP_CLOSE = 1.0, 0.08
SLIDE_MIN, SLIDE_MAX = -0.04, 0.87

# Safety limits / achievable envelope.  The server's base responds to /cmd_vel
# at roughly 1/8 of the commanded value (measured), so a publish-time gain maps
# the physical command into the responsive range without saturating the wheel PID.
MAX_LIN, MAX_ANG = 0.20, 0.35
MAX_LIN_ACC, MAX_ANG_ACC = 0.5, 1.5
JOINT_SLEW = 1.2  # rad/s for the fastest joint

BASE_GAIN_LIN = 8.0
BASE_GAIN_ANG = 4.0


def wrap_to_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class MMK2Adapter(Node):
    def __init__(self, node_name: str = "competition_mmk2_adapter", rate_hz: float = 50.0):
        super().__init__(node_name)
        self.kdl = MMK2Kdl()

        self.tc = np.zeros(19)
        self.tc[5:11] = INIT_ARM_L
        self.tc[11] = GRIP_OPEN
        self.tc[12:18] = INIT_ARM_R
        self.tc[18] = GRIP_OPEN

        self.action = self.tc.copy()
        self.tc_prev = self.tc.copy()
        self.joint_move_ratio = np.ones(19)

        self.dt = 1.0 / rate_hz
        self.max_lin = MAX_LIN
        self.max_ang = MAX_ANG
        self.des_lin = self.des_ang = 0.0
        self.cur_lin = self.cur_ang = 0.0

        self.base_xy: Optional[np.ndarray] = None
        self.base_yaw = 0.0
        self.jpos: dict = {}
        self.jvel: dict = {}

        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 5)
        self.spine_pub = self.create_publisher(
            Float64MultiArray, "/spine_forward_position_controller/commands", 5)
        self.head_pub = self.create_publisher(
            Float64MultiArray, "/head_forward_position_controller/commands", 5)
        self.larm_pub = self.create_publisher(
            Float64MultiArray, "/left_arm_forward_position_controller/commands", 5)
        self.rarm_pub = self.create_publisher(
            Float64MultiArray, "/right_arm_forward_position_controller/commands", 5)

        self.create_subscription(
            Odometry, "/slamware_ros_sdk_server_node/odom", self._odom_cb, 10)
        self.create_subscription(JointState, "/joint_states", self._js_cb, 10)

    # ---- feedback ----
    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.base_xy = np.array([p.x, p.y])
        self.base_yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]

    def _js_cb(self, msg: JointState) -> None:
        self.jpos = {n: msg.position[i] for i, n in enumerate(msg.name)
                     if i < len(msg.position)}
        self.jvel = {n: msg.velocity[i] for i, n in enumerate(msg.name)
                     if i < len(msg.velocity)}

    @property
    def ready(self) -> bool:
        return self.base_xy is not None and bool(self.jpos)

    @property
    def slide_meas(self) -> float:
        return float(self.jpos.get("slide_joint", self.tc[2]))

    def arm_meas(self, side: str) -> np.ndarray:
        prefix = "left" if side == "left" else "right"
        base = 5 if side == "left" else 12
        return np.array([
            self.jpos.get(f"{prefix}_arm_joint{i + 1}", self.tc[base + i])
            for i in range(6)
        ])

    @property
    def rarm_meas(self) -> np.ndarray:
        return self.arm_meas("right")

    # ---- command setters ----
    def set_base_velocity(self, lin: float, ang: float) -> None:
        self.des_lin = float(np.clip(lin, -self.max_lin, self.max_lin))
        self.des_ang = float(np.clip(ang, -self.max_ang, self.max_ang))

    def stop_base(self) -> None:
        self.set_base_velocity(0.0, 0.0)

    def set_slide(self, value: float) -> None:
        self.tc[2] = float(np.clip(value, SLIDE_MIN, SLIDE_MAX))

    def set_head(self, yaw: float, pitch: float) -> None:
        self.tc[3] = float(yaw)
        self.tc[4] = float(pitch)

    def set_gripper(self, side: str, value: float) -> None:
        self.tc[11 if side == "left" else 18] = float(value)

    def set_arm_joints(self, side: str, joints) -> None:
        base = 5 if side == "left" else 12
        self.tc[base:base + 6] = np.asarray(joints, dtype=float)[:6]

    def home(self) -> None:
        """Safe pose: arms tucked to init, grippers open, slide low, head level."""
        self.tc[2] = 0.0
        self.tc[3] = 0.0
        self.tc[4] = 0.0
        self.set_arm_joints("left", INIT_ARM_L)
        self.set_arm_joints("right", INIT_ARM_R)
        self.tc[11] = GRIP_OPEN
        self.tc[18] = GRIP_OPEN

    # ---- IK helper (right arm), footprint frame ----
    def world_to_footprint(self, p_world) -> np.ndarray:
        d = np.asarray(p_world, dtype=float) - np.array(
            [self.base_xy[0], self.base_xy[1], 0.0])
        c, s = math.cos(-self.base_yaw), math.sin(-self.base_yaw)
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]])

    def footprint_to_world(self, fp) -> np.ndarray:
        c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
        fp = np.asarray(fp, dtype=float)
        return np.array([self.base_xy[0] + c * fp[0] - s * fp[1],
                         self.base_xy[1] + s * fp[0] + c * fp[1], fp[2]])

    def arm_to(self, side: str, world_pos, rot=None) -> bool:
        """Solve IK for ``side`` arm to reach ``world_pos`` (footprint frame)."""
        if rot is None:
            rot = np.eye(3)
        fp = self.world_to_footprint(world_pos)
        T = np.eye(4)
        T[:3, :3] = rot
        T[:3, 3] = fp
        ref = np.zeros(7)
        ref[0] = float(self.tc[2])
        ref[1:] = self.arm_meas(side)
        if side == "right":
            sols = self.kdl.inverse_kinematics(
                T_left=None, T_right=T, ref_pos=ref, target_height=float(self.tc[2]))
        else:
            sols = self.kdl.inverse_kinematics(
                T_left=T, T_right=None, ref_pos=ref, target_height=float(self.tc[2]))
        if sols:
            self.set_arm_joints(side, np.asarray(sols[0])[1:7])
            return True
        self.get_logger().warn(
            f"IK unreachable ({side}): world={np.round(world_pos, 3)}")
        return False

    def ee_world(self, side: str = "right") -> np.ndarray:
        q = np.concatenate([[self.slide_meas], self.arm_meas(side)])
        _, T = self.kdl.forward_kinematics(q, index=side)
        return self.footprint_to_world(T[:3, 3])

    # ---- step / publish ----
    def _ramp_twist(self) -> None:
        dl = np.clip(self.des_lin - self.cur_lin, -MAX_LIN_ACC * self.dt, MAX_LIN_ACC * self.dt)
        da = np.clip(self.des_ang - self.cur_ang, -MAX_ANG_ACC * self.dt, MAX_ANG_ACC * self.dt)
        self.cur_lin += dl
        self.cur_ang += da
        self.tc[0], self.tc[1] = self.cur_lin, self.cur_ang

    def _smooth_step(self) -> None:
        if not np.allclose(self.tc[2:19], self.tc_prev[2:19]):
            dif = np.abs(self.action[2:19] - self.tc[2:19])
            self.joint_move_ratio[2:19] = dif / (np.max(dif) + 1e-6)
            self.joint_move_ratio[2] *= 0.3
            self.tc_prev[:] = self.tc
        step = JOINT_SLEW * self.dt
        for i in range(2, 19):
            self.action[i] = step_func(
                self.action[i], self.tc[i], self.joint_move_ratio[i] * step)

    def publish(self) -> None:
        tw = Twist()
        tw.linear.x = float(self.tc[0]) * BASE_GAIN_LIN
        tw.angular.z = float(self.tc[1]) * BASE_GAIN_ANG
        self.cmd_vel_pub.publish(tw)
        self.spine_pub.publish(Float64MultiArray(data=[float(self.action[2])]))
        self.head_pub.publish(Float64MultiArray(
            data=[float(self.action[3]), float(self.action[4])]))
        self.larm_pub.publish(Float64MultiArray(
            data=[float(x) for x in self.action[5:11]] + [float(self.action[11])]))
        self.rarm_pub.publish(Float64MultiArray(
            data=[float(x) for x in self.action[12:18]] + [float(self.action[18])]))

    def step(self) -> None:
        self._ramp_twist()
        self._smooth_step()
        self.publish()

    def emergency_stop(self) -> None:
        """Immediately command zero base velocity and publish the current pose."""
        self.des_lin = self.des_ang = 0.0
        self.cur_lin = self.cur_ang = 0.0
        self.tc[0] = self.tc[1] = 0.0
        self.publish()
