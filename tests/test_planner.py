import math
import unittest

import numpy as np

from competition_client.planner import GridPlanner


class ScanProjectionTests(unittest.TestCase):
    def test_projects_hit_with_live_sensor_pose(self):
        planner = GridPlanner()
        planner.update_scan(
            np.array([1.0]), 0.0, 1.0, 0.0, 0.0, 0.0,
            sensor_pose=(0.1, 0.2, math.pi / 2.0),
            range_min=0.02, range_max=12.0,
        )

        hit = planner.world_to_idx(0.1, 1.2)
        wrong_unrotated_hit = planner.world_to_idx(1.1, 0.2)
        self.assertTrue(planner.dynamic[hit])
        self.assertFalse(planner.dynamic[wrong_unrotated_hit])

    def test_ignores_echo_outside_message_range(self):
        planner = GridPlanner()
        planner.update_scan(
            np.array([0.5]), 0.0, 1.0, 0.0, 0.0, 0.0,
            sensor_pose=(0.0, 0.0, 0.0),
            range_min=0.8, range_max=12.0,
        )

        hit = planner.world_to_idx(0.5, 0.0)
        self.assertFalse(planner.dynamic[hit])

    def test_drops_self_echo_behind_lidar_face(self):
        planner = GridPlanner()
        planner.update_scan(
            np.array([0.2]), math.pi, 1.0, 0.0, 0.0, 0.0,
            sensor_pose=(0.1137, 0.0, 0.0),
            range_min=0.02, range_max=12.0,
        )

        behind = planner.world_to_idx(0.1137 - 0.2, 0.0)
        self.assertFalse(planner.dynamic[behind])

    def test_keeps_close_obstacle_in_front_of_lidar_face(self):
        planner = GridPlanner()
        planner.update_scan(
            np.array([0.12]), 0.0, 1.0, 0.0, 0.0, 0.0,
            sensor_pose=(0.1137, 0.0, 0.0),
            range_min=0.02, range_max=12.0,
        )

        front = planner.world_to_idx(0.1137 + 0.12, 0.0)
        self.assertTrue(planner.dynamic[front])


if __name__ == "__main__":
    unittest.main()
