#!/usr/bin/env python3
"""Shelf inventory: build a kind -> candidate slots map from perception.

Only camera + ArUco observations feed this map; it never reads server truth.
The map is valid only for the current ``run_prefix`` and must be reset when a
new task arrives.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SlotKey = Tuple[str, str, str]  # (shelf, level, column)


@dataclass
class SlotRecord:
    kind: str
    slot: SlotKey
    world: List[float]
    confidence: float
    aruco_id: Optional[int] = None
    last_seen: float = field(default_factory=time.time)
    hits: int = 1

    @property
    def key(self) -> SlotKey:
        return self.slot


class ShelfInventory:
    def __init__(self, min_confidence: float = 0.35):
        self.min_confidence = min_confidence
        self._records: Dict[SlotKey, SlotRecord] = {}

    def reset(self) -> None:
        self._records.clear()

    def update(self, products: List[dict], arucos: Optional[List[dict]] = None) -> None:
        aruco_by_slot = {}
        if arucos:
            for a in arucos:
                slot = self._slot_of(a.get("world"))
                if slot is not None:
                    aruco_by_slot[slot] = a.get("id")

        now = time.time()
        for p in products:
            if p.get("conf", 0.0) < self.min_confidence:
                continue
            slot = None
            s = p.get("slot")
            if s:
                slot = (s["shelf"], s["level"], s["column"])
            if slot is None:
                slot = self._slot_of(p.get("world"))
            if slot is None:
                continue
            prev = self._records.get(slot)
            if prev is not None and prev.kind == p["kind"] and prev.confidence >= p["conf"]:
                prev.last_seen = now
                prev.hits += 1
                continue
            self._records[slot] = SlotRecord(
                kind=p["kind"],
                slot=slot,
                world=list(p.get("world", [0, 0, 0])),
                confidence=float(p["conf"]),
                aruco_id=p.get("aruco_id") or aruco_by_slot.get(slot),
                last_seen=now,
                hits=1 + (prev.hits if prev and prev.kind == p["kind"] else 0),
            )

    @staticmethod
    def _slot_of(world) -> Optional[SlotKey]:
        if not world:
            return None
        try:
            from competition_client.perception import slot_from_world
        except Exception:
            from perception import slot_from_world
        s = slot_from_world(world)
        if s is None:
            return None
        return (s["shelf"], s["level"], s["column"])

    def candidates(self, kind: str) -> List[SlotRecord]:
        out = [r for r in self._records.values() if r.kind == kind]
        out.sort(key=lambda r: (-r.confidence, -r.hits))
        return out

    def all(self) -> List[SlotRecord]:
        return list(self._records.values())

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self._records.values():
            counts[r.kind] = counts.get(r.kind, 0) + 1
        return counts
