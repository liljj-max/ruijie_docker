#!/usr/bin/env python3
"""Measure the base angular-velocity response to /cmd_vel (debug helper)."""
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy.spatial.transform import Rotation


def main():
    rclpy.init()
    node = Node("angtest")
    pub = node.create_publisher(Twist, "/cmd_vel", 10)
    state = {"yaw": None, "xy": None}

    def cb(msg):
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        state["yaw"] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
        state["xy"] = (p.x, p.y)

    node.create_subscription(Odometry, "/slamware_ros_sdk_server_node/odom", cb, 10)

    for w in (0.5, 1.0, 1.2, 2.0, 4.0, 6.0):
        t = Twist()
        t.angular.z = float(w)
        t0 = None
        end = time.time() + 3.0
        while time.time() < end and rclpy.ok():
            pub.publish(t)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.05)
            if t0 is None and state["yaw"] is not None:
                t0 = (time.time(), state["yaw"])
        if t0:
            dt = time.time() - t0[0]
            print(f"ang cmd={w:.2f} -> measured {abs(state['yaw'] - t0[1]) / dt:.3f} rad/s")
        t.angular.z = 0.0
        for _ in range(10):
            pub.publish(t)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.05)

    for v in (0.45, 1.0, 2.0, 3.0):
        t = Twist()
        t.linear.x = float(v)
        p0 = None
        end = time.time() + 3.0
        while time.time() < end and rclpy.ok():
            pub.publish(t)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.05)
            if p0 is None and state["xy"] is not None:
                p0 = (time.time(), state["xy"])
        if p0:
            dt = time.time() - p0[0]
            dist = math.hypot(state["xy"][0] - p0[1][0], state["xy"][1] - p0[1][1])
            print(f"lin cmd={v:.2f} -> measured {dist / dt:.3f} m/s")
        t.linear.x = 0.0
        for _ in range(10):
            pub.publish(t)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.05)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
