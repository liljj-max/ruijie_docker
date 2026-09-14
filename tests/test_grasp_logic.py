import unittest

import numpy as np

from competition_client.grasp_logic import (
    StableTargetTracker,
    base_is_stopped,
    creep_speed,
    joints_are_settled,
)


class StableTargetTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = StableTargetTracker(
            min_frames=5,
            max_frames=8,
            max_xy_std=0.015,
            max_z_std=0.020,
        )
        self.target_slot = ("C", "L2", "C2")
        self.to_footprint = lambda p: np.array([0.6, 0.0, p[2]])

    @staticmethod
    def _det(stamp, world, slot=("C", "L2", "C2"), kind="kele", conf=0.9):
        return {
            "stamp": stamp,
            "kind": kind,
            "conf": conf,
            "world": list(world),
            "slot": {"shelf": slot[0], "level": slot[1], "column": slot[2]},
            "aruco_id": 22,
        }

    def test_requires_distinct_frames_from_the_selected_slot(self):
        wrong_slot = self._det(1.0, [0.4, 3.3, 0.85], slot=("D", "L2", "C2"))
        correct = self._det(1.0, [0.03, 3.32, 0.85])

        self.assertIsNone(self.tracker.add_frame(
            1.0, [wrong_slot, correct], "kele", self.target_slot, self.to_footprint))
        self.assertEqual(self.tracker.sample_count, 1)

        # Reprocessing the same perception frame must not count as a new sample.
        self.assertIsNone(self.tracker.add_frame(
            1.0, [correct], "kele", self.target_slot, self.to_footprint))
        self.assertEqual(self.tracker.sample_count, 1)

        result = None
        for i, x in enumerate((0.031, 0.029, 0.032, 0.030), start=2):
            result = self.tracker.add_frame(
                float(i), [self._det(float(i), [x, 3.32, 0.85])],
                "kele", self.target_slot, self.to_footprint)

        self.assertIsNotNone(result)
        np.testing.assert_allclose(result, [0.03, 3.32, 0.85], atol=0.002)

    def test_rejects_a_spatially_unstable_lock(self):
        result = None
        for i, x in enumerate((0.00, 0.05, -0.04, 0.06, -0.05), start=1):
            result = self.tracker.add_frame(
                float(i), [self._det(float(i), [x, 3.32, 0.85])],
                "kele", self.target_slot, self.to_footprint)

        self.assertIsNone(result)
        self.assertEqual(self.tracker.sample_count, 5)

    def test_rejects_points_outside_the_grasp_envelope(self):
        for i in range(1, 7):
            self.tracker.add_frame(
                float(i), [self._det(float(i), [0.6, 0.30, 0.85])],
                "kele", self.target_slot, lambda p: p)

        self.assertEqual(self.tracker.sample_count, 0)

    def test_rejects_low_confidence_detections(self):
        for i in range(1, 7):
            self.tracker.add_frame(
                float(i), [self._det(float(i), [0.03, 3.32, 0.85], conf=0.40)],
                "kele", self.target_slot, self.to_footprint)

        self.assertEqual(self.tracker.sample_count, 0)


class CreepSpeedTests(unittest.TestCase):
    def test_stops_inside_the_grasp_gap(self):
        self.assertEqual(creep_speed(0.030, stop_gap=0.035), 0.0)

    def test_slows_down_near_the_object(self):
        near = creep_speed(0.055, stop_gap=0.035)
        far = creep_speed(0.20, stop_gap=0.035)

        self.assertGreaterEqual(near, 0.01)
        self.assertLess(near, far)
        self.assertLessEqual(far, 0.03)


class SettlingTests(unittest.TestCase):
    def test_arm_requires_both_position_and_velocity_to_settle(self):
        target = np.array([0.2, -0.3, 0.4])

        self.assertTrue(joints_are_settled(
            target, [0.21, -0.31, 0.39], [0.01, 0.02, 0.01],
            position_tolerance=0.03, velocity_tolerance=0.05))
        self.assertFalse(joints_are_settled(
            target, [0.21, -0.31, 0.39], [0.01, 0.08, 0.01],
            position_tolerance=0.03, velocity_tolerance=0.05))
        self.assertFalse(joints_are_settled(
            target, [0.21, -0.36, 0.39], [0.01, 0.02, 0.01],
            position_tolerance=0.03, velocity_tolerance=0.05))

    def test_base_must_stop_translating_and_rotating(self):
        self.assertTrue(base_is_stopped(0.005, 0.01))
        self.assertFalse(base_is_stopped(0.02, 0.01))
        self.assertFalse(base_is_stopped(0.005, 0.05))


if __name__ == "__main__":
    unittest.main()
