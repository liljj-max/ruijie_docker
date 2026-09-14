#!/usr/bin/env python3
"""P1 smoke test: receive the official task and exercise the MMK2 adapter.

Run inside the official Client container:
    cd /workspace/baseline && python3 -m scripts.task_probe
"""

import sys
import time
from pathlib import Path

BASELINE_ROOT = Path(__file__).resolve().parents[1]
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

import rclpy
from rclpy.executors import SingleThreadedExecutor

from competition_client.mmk2_adapter import MMK2Adapter
from competition_client.task_parser import TaskListener


def main() -> int:
    rclpy.init()
    state = {}

    def on_new_task(task):
        state["task"] = task

    listener = TaskListener(on_new_task=on_new_task)
    adapter = MMK2Adapter()

    executor = SingleThreadedExecutor()
    executor.add_node(listener)
    executor.add_node(adapter)

    deadline = time.time() + 8.0
    while rclpy.ok() and time.time() < deadline:
        executor.spin_once(timeout_sec=0.1)

    task = state.get("task")
    if task is None:
        print("[probe] NO TASK RECEIVED")
        rc = 1
    else:
        print(f"[probe] task run_prefix={task.run_prefix} count={task.count}")
        for t in task.targets:
            print(f"    id={t.id} kind={t.kind}")
        rc = 0

    print(f"[probe] adapter ready={adapter.ready} base_xy={adapter.base_xy} "
          f"slide={adapter.slide_meas:.3f}")
    if adapter.ready:
        print(f"[probe] base_yaw={adapter.base_yaw:.3f}")

    print("[probe] -> home pose, publish 2s")
    adapter.stop_base()
    adapter.home()
    end = time.time() + 2.0
    while rclpy.ok() and time.time() < end:
        adapter.step()
        executor.spin_once(timeout_sec=0.0)
        time.sleep(adapter.dt)

    print("[probe] -> emergency stop")
    adapter.emergency_stop()

    executor.remove_node(listener)
    executor.remove_node(adapter)
    listener.destroy_node()
    adapter.destroy_node()
    rclpy.shutdown()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
