import unittest

import numpy as np

from competition_client.perception import PerceptionNode, validated_aruco_anchors
from competition_client.shelf_scanner import ShelfInventory


SLOT = {"shelf": "A", "level": "L1", "column": "C1"}
SLOT_KEY = ("A", "L1", "C1")


def product(kind="kele", conf=0.9, aruco_id=0, world=None):
    return {
        "kind": kind,
        "conf": conf,
        "world": world or [-1.955, 3.323, 0.499],
        "slot": dict(SLOT),
        "aruco_id": aruco_id,
        "stamp": 1.0,
        "slot_source": "aruco",
    }


class ShelfInventoryTests(unittest.TestCase):
    def test_preserves_aruco_id_zero(self):
        inventory = ShelfInventory()
        inventory.update([product(aruco_id=0)], [{"id": 17, "world": product()["world"]}])

        self.assertEqual(inventory.candidates("kele")[0].aruco_id, 0)

    def test_single_different_low_confidence_frame_does_not_replace_stable_kind(self):
        inventory = ShelfInventory()
        for x in (-1.95, -1.96, -1.94):
            inventory.update([product(world=[x, 3.323, 0.499])])
        inventory.update([product(kind="shupian", conf=0.36, world=[-1.5, 3.0, 0.8])])

        record = inventory.candidates("kele")[0]
        self.assertEqual(record.kind, "kele")
        self.assertEqual(record.hits, 3)
        np.testing.assert_allclose(record.world, [-1.95, 3.323, 0.499])

    def test_consumed_or_reserved_slots_are_not_candidates(self):
        inventory = ShelfInventory()
        inventory.update([product()])
        self.assertTrue(inventory.reserve(SLOT_KEY))
        self.assertEqual(inventory.candidates("kele"), [])
        inventory.release(SLOT_KEY)
        self.assertTrue(inventory.consume(SLOT_KEY))
        self.assertFalse(inventory.is_available(SLOT_KEY))
        self.assertEqual(inventory.candidates("kele"), [])


class AssociationTests(unittest.TestCase):
    def test_rejects_marker_whose_id_disagrees_with_world_slot(self):
        markers = [
            {"id": 0, "world": [-1.955, 3.323, 0.499]},
            {"id": 1, "world": [-1.955, 3.323, 0.499]},
        ]

        anchors = validated_aruco_anchors(markers)

        self.assertEqual([anchor[2] for anchor in anchors], [0])

    def test_rejects_ambiguous_markers(self):
        anchors = [
            (("A", "L1", "C1"), np.array([0.0, 0.0, 0.0]), 0),
            (("A", "L1", "C2"), np.array([0.2, 0.0, 0.0]), 1),
        ]

        slot, aruco_id, source = PerceptionNode._associate(
            np.array([0.1, 0.0, 0.0]), anchors)

        self.assertIsNone(aruco_id)
        self.assertNotEqual(source, "aruco")

    def test_marker_is_one_to_one_and_distance_is_strict(self):
        anchors = [(('A', 'L1', 'C1'), np.array([0.0, 0.0, 0.0]), 0)]
        self.assertEqual(
            PerceptionNode._associate([0.1, 0.0, 0.0], anchors, set())[1], 0)
        self.assertIsNone(
            PerceptionNode._associate([0.1, 0.0, 0.0], anchors, {0})[1])
        self.assertIsNone(
            PerceptionNode._associate([0.22, 0.0, 0.0], anchors, set())[1])


if __name__ == "__main__":
    unittest.main()
