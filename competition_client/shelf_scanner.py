#!/usr/bin/env python3
"""Shelf inventory: build a kind -> candidate slots map from perception.

Only camera + ArUco observations feed this map; it never reads server truth.
The map is valid only for the current ``run_prefix`` and must be reset when a
new task arrives.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from statistics import median
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


@dataclass
class _KindEvidence:
    confidences: deque = field(default_factory=lambda: deque(maxlen=12))
    worlds: deque = field(default_factory=lambda: deque(maxlen=9))
    aruco_id: Optional[int] = None
    last_seen: float = 0.0

    @property
    def hits(self) -> int:
        return len(self.confidences)

    @property
    def confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences)


class ShelfInventory:
    def __init__(self, min_confidence: float = 0.35):
        self.min_confidence = min_confidence
        self._records: Dict[SlotKey, SlotRecord] = {}
        self._evidence: Dict[SlotKey, Dict[str, _KindEvidence]] = {}
        self._reserved = set()
        self._consumed = set()

    def reset(self) -> None:
        self._records.clear()
        self._evidence.clear()
        self._reserved.clear()
        self._consumed.clear()

    def update(self, products: List[dict], arucos: Optional[List[dict]] = None) -> None:
        # Kept in the signature for callers that still pass it.  Product identity
        # and slot metadata must come from one perception frame, never this list.
        del arucos
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
            kind = p.get("kind")
            if not kind:
                continue
            evidence_by_kind = self._evidence.setdefault(slot, {})
            evidence = evidence_by_kind.setdefault(kind, _KindEvidence())
            evidence.confidences.append(float(p["conf"]))
            world = p.get("world")
            if world is not None and len(world) >= 3:
                evidence.worlds.append(tuple(float(v) for v in world[:3]))
            if p.get("aruco_id") is not None:
                evidence.aruco_id = int(p["aruco_id"])
            evidence.last_seen = now

            winner_kind, winner = max(
                evidence_by_kind.items(),
                key=lambda item: (item[1].hits, item[1].confidence,
                                  item[1].last_seen))
            robust_world = ([median(point[i] for point in winner.worlds)
                             for i in range(3)] if winner.worlds else [0.0, 0.0, 0.0])
            self._records[slot] = SlotRecord(
                kind=winner_kind,
                slot=slot,
                world=robust_world,
                confidence=winner.confidence,
                aruco_id=winner.aruco_id,
                last_seen=winner.last_seen,
                hits=winner.hits,
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
        out = [r for r in self._records.values()
               if r.kind == kind and self.is_available(r.slot)]
        out.sort(key=lambda r: (-r.hits, -r.confidence))
        return out

    def reserve(self, slot: SlotKey) -> bool:
        if not self.is_available(slot):
            return False
        self._reserved.add(slot)
        return True

    def consume(self, slot: SlotKey) -> bool:
        if slot not in self._records or slot in self._consumed:
            return False
        self._reserved.discard(slot)
        self._consumed.add(slot)
        return True

    def release(self, slot: SlotKey) -> None:
        self._reserved.discard(slot)

    def is_available(self, slot: SlotKey) -> bool:
        return (slot in self._records and slot not in self._reserved
                and slot not in self._consumed)

    def all(self) -> List[SlotRecord]:
        return list(self._records.values())

    def summary(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for r in self._records.values():
            counts[r.kind] = counts.get(r.kind, 0) + 1
        return counts
