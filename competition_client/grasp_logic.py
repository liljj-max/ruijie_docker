"""Pure helpers for stable target locking and slow final approach."""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np


SlotKey = Tuple[str, str, str]


def _slot_key(value) -> Optional[SlotKey]:
    if isinstance(value, dict):
        try:
            return value["shelf"], value["level"], value["column"]
        except KeyError:
            return None
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return str(value[0]), str(value[1]), str(value[2])
    return None


class StableTargetTracker:
    """Keep at most one selected-target observation per perception frame."""

    def __init__(
        self,
        min_frames: int = 5,
        max_frames: int = 12,
        max_xy_std: float = 0.015,
        max_z_std: float = 0.020,
        min_confidence: float = 0.50,
        reach_fwd: Tuple[float, float] = (0.3, 1.5),
        reach_lateral: float = 0.16,
        reach_z: Tuple[float, float] = (0.40, 1.35),
    ):
        self.min_frames = int(min_frames)
        self.max_xy_std = float(max_xy_std)
        self.max_z_std = float(max_z_std)
        self.min_confidence = float(min_confidence)
        self.reach_fwd = reach_fwd
        self.reach_lateral = float(reach_lateral)
        self.reach_z = reach_z
        self._samples = deque(maxlen=int(max_frames))
        self._seen_stamps = deque(maxlen=max(32, int(max_frames) * 4))

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def clear(self) -> None:
        self._samples.clear()
        self._seen_stamps.clear()

    def add_frame(
        self,
        stamp: float,
        detections: Iterable[dict],
        target_kind: str,
        target_slot: SlotKey,
        world_to_footprint: Callable[[np.ndarray], Sequence[float]],
        expected_aruco_id: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        stamp = float(stamp)
        if stamp in self._seen_stamps:
            return self.stable_point()
        self._seen_stamps.append(stamp)

        candidates = []
        for detection in detections:
            if detection.get("kind") != target_kind:
                continue
            if float(detection.get("conf", 0.0)) < self.min_confidence:
                continue
            if _slot_key(detection.get("slot")) != target_slot:
                continue
            marker_id = detection.get("aruco_id")
            if (expected_aruco_id is not None and marker_id is not None
                    and int(marker_id) != int(expected_aruco_id)):
                continue
            try:
                world = np.asarray(detection["world"], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            if world.shape != (3,) or not np.all(np.isfinite(world)):
                continue
            footprint = np.asarray(world_to_footprint(world), dtype=float)
            if footprint.shape != (3,) or not np.all(np.isfinite(footprint)):
                continue
            if not (self.reach_fwd[0] <= footprint[0] <= self.reach_fwd[1]):
                continue
            if abs(footprint[1]) > self.reach_lateral:
                continue
            if not (self.reach_z[0] <= world[2] <= self.reach_z[1]):
                continue
            candidates.append((float(detection.get("conf", 0.0)), world))

        if candidates:
            _, best_world = max(candidates, key=lambda item: item[0])
            self._samples.append(best_world)
        return self.stable_point()

    def stable_point(self) -> Optional[np.ndarray]:
        if len(self._samples) < self.min_frames:
            return None
        points = np.asarray(self._samples, dtype=float)
        std = np.std(points, axis=0)
        if max(float(std[0]), float(std[1])) > self.max_xy_std:
            return None
        if float(std[2]) > self.max_z_std:
            return None
        return np.median(points, axis=0)


def creep_speed(
    remaining: float,
    stop_gap: float = 0.035,
    min_speed: float = 0.01,
    max_speed: float = 0.03,
    slow_distance: float = 0.15,
) -> float:
    """Return a distance-proportional speed for the final grasp approach."""
    error = float(remaining) - float(stop_gap)
    if error <= 0.0:
        return 0.0
    ratio = min(1.0, error / max(float(slow_distance), 1e-6))
    return float(min_speed + (max_speed - min_speed) * ratio)


def joints_are_settled(
    target: Sequence[float],
    measured: Sequence[float],
    velocities: Sequence[float],
    position_tolerance: float,
    velocity_tolerance: float,
) -> bool:
    """Return True when every joint is both near target and nearly stationary."""
    target_arr = np.asarray(target, dtype=float)
    measured_arr = np.asarray(measured, dtype=float)
    velocity_arr = np.asarray(velocities, dtype=float)
    if target_arr.shape != measured_arr.shape or target_arr.shape != velocity_arr.shape:
        return False
    if not (np.all(np.isfinite(target_arr)) and np.all(np.isfinite(measured_arr))
            and np.all(np.isfinite(velocity_arr))):
        return False
    return bool(
        np.max(np.abs(target_arr - measured_arr)) <= position_tolerance
        and np.max(np.abs(velocity_arr)) <= velocity_tolerance
    )


def base_is_stopped(
    linear_speed: float,
    angular_speed: float,
    linear_tolerance: float = 0.01,
    angular_tolerance: float = 0.03,
) -> bool:
    """Return True when measured base motion is below grasping thresholds."""
    return (abs(float(linear_speed)) <= linear_tolerance
            and abs(float(angular_speed)) <= angular_tolerance)
