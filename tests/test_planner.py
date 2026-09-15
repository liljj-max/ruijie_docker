import unittest

import numpy as np

from competition_client.planner import LASER_OFFSET, GridPlanner


class ScanProjectionTests(unittest.TestCase):
    def test_projects_hit_with_laser_offset(self):
        planner = GridPlanner()
        planner.update_scan(np.array([1.0]), 0.0, 1.0, 0.0, 0.0, 0.0)

        hit = planner.world_to_idx(LASER_OFFSET[0] + 1.0, 0.0)
        wrong = planner.world_to_idx(1.0, 0.0)
        self.assertTrue(planner.dynamic[hit])
        self.assertFalse(planner.dynamic[wrong])

    def test_ignores_echo_below_min_range(self):
        planner = GridPlanner()
        planner.update_scan(np.array([0.1]), 0.0, 1.0, 0.0, 0.0, 0.0)

        hit = planner.world_to_idx(LASER_OFFSET[0] + 0.1, 0.0)
        self.assertFalse(planner.dynamic[hit])


if __name__ == "__main__":
    unittest.main()
