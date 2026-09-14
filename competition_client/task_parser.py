#!/usr/bin/env python3
"""Task parsing for /supermarket_sorting/task.

The server publishes the whole task list once on a TRANSIENT_LOCAL (latched)
topic, so a late-joining client still receives it.  A new ``run_prefix`` means a
new run: callers must drop any cached inventory/progress from the previous run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String

DEFAULT_TASK_TOPIC = "/supermarket_sorting/task"


@dataclass(frozen=True)
class TaskTarget:
    id: str
    kind: str


@dataclass
class Task:
    schema_version: int
    run_prefix: str
    count: int
    targets: List[TaskTarget] = field(default_factory=list)

    def kinds(self) -> List[str]:
        return [t.kind for t in self.targets]


def parse_task(raw: str) -> Task:
    """Parse a task JSON string. Raises ValueError on malformed input."""
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("task payload is not a JSON object")

    schema_version = int(data.get("schema_version", 0))
    run_prefix = str(data.get("run_prefix", ""))
    raw_targets = data.get("targets", [])
    if not isinstance(raw_targets, list):
        raise ValueError("targets is not a list")

    targets: List[TaskTarget] = []
    for item in raw_targets:
        if not isinstance(item, dict):
            continue
        tid = item.get("id")
        kind = item.get("kind")
        if tid is None or kind is None:
            continue
        targets.append(TaskTarget(id=str(tid), kind=str(kind)))

    count = int(data.get("count", len(targets)))
    return Task(
        schema_version=schema_version,
        run_prefix=run_prefix,
        count=count,
        targets=targets,
    )


class TaskListener(Node):
    """Subscribe to the latched task topic and expose the latest Task.

    ``on_new_task`` is invoked once per distinct ``run_prefix`` (including the
    first one).  This is the single place that decides a new run has started.
    """

    def __init__(
        self,
        node_name: str = "competition_task_listener",
        topic: str = DEFAULT_TASK_TOPIC,
        on_new_task: Optional[Callable[[Task], None]] = None,
    ):
        super().__init__(node_name)
        self._topic = topic
        self._on_new_task = on_new_task
        self.task: Optional[Task] = None

        qos = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._sub = self.create_subscription(String, topic, self._cb, qos)
        self.get_logger().info(f"task listener up on {topic}")

    def _cb(self, msg: String) -> None:
        try:
            task = parse_task(msg.data)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"task parse failed: {exc}")
            return

        is_new = self.task is None or task.run_prefix != self.task.run_prefix
        self.task = task
        if is_new:
            self.get_logger().info(
                "new task: run_prefix=%s count=%d kinds=%s"
                % (task.run_prefix, task.count, task.kinds())
            )
            if self._on_new_task is not None:
                self._on_new_task(task)
