#!/usr/bin/env python3

import math
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan


class LaneToScanNode(Node):
    def __init__(self) -> None:
        super().__init__("lane_to_scan_node")

        # ---------------- Topics ----------------
        self.declare_parameter("left_path_topic", "/yolopv2/left_lane_path")
        self.declare_parameter("right_path_topic", "/yolopv2/right_lane_path")
        self.declare_parameter("objects_path_topic", "/yolo/objects_world_path")
        self.declare_parameter("scan_topic", "/yolopv2/lane_scan")

        # ---------------- Scan settings ----------------
        self.declare_parameter("angle_min", -1.57)
        self.declare_parameter("angle_max", 1.57)
        self.declare_parameter("angle_increment", 0.01)
        self.declare_parameter("range_min", 0.05)
        self.declare_parameter("range_max", 30.0)

        # ---------------- Behavior ----------------
        self.declare_parameter("publish_on_each_update", True)
        self.declare_parameter("use_inf", True)
        self.declare_parameter("output_frame", "")  # empty = use incoming frame_id
        self.declare_parameter("keep_closest_only", True)

        self.left_path_topic = str(self.get_parameter("left_path_topic").value)
        self.right_path_topic = str(self.get_parameter("right_path_topic").value)
        self.objects_path_topic = str(self.get_parameter("objects_path_topic").value)
        self.scan_topic = str(self.get_parameter("scan_topic").value)

        self.angle_min = float(self.get_parameter("angle_min").value)
        self.angle_max = float(self.get_parameter("angle_max").value)
        self.angle_increment = float(self.get_parameter("angle_increment").value)
        self.range_min = float(self.get_parameter("range_min").value)
        self.range_max = float(self.get_parameter("range_max").value)

        self.publish_on_each_update = bool(self.get_parameter("publish_on_each_update").value)
        self.use_inf = bool(self.get_parameter("use_inf").value)
        self.output_frame = str(self.get_parameter("output_frame").value)
        self.keep_closest_only = bool(self.get_parameter("keep_closest_only").value)

        if self.angle_increment <= 0.0:
            raise ValueError("angle_increment must be > 0")
        if self.angle_max <= self.angle_min:
            raise ValueError("angle_max must be > angle_min")
        if self.range_max <= self.range_min:
            raise ValueError("range_max must be > range_min")

        self.pub_scan = self.create_publisher(LaserScan, self.scan_topic, 10)

        self.sub_left = self.create_subscription(Path, self.left_path_topic, self.cb_left, 10)
        self.sub_right = self.create_subscription(Path, self.right_path_topic, self.cb_right, 10)
        self.sub_objects = self.create_subscription(Path, self.objects_path_topic, self.cb_objects, 10)

        self.left_msg: Optional[Path] = None
        self.right_msg: Optional[Path] = None
        self.objects_msg: Optional[Path] = None

        self.get_logger().info(f"Subscribed left path:    {self.left_path_topic}")
        self.get_logger().info(f"Subscribed right path:   {self.right_path_topic}")
        self.get_logger().info(f"Subscribed objects path: {self.objects_path_topic}")
        self.get_logger().info(f"Publishing scan:         {self.scan_topic}")
        self.get_logger().info("Using XY plane: x=forward, y=lateral, z=0")

    def cb_left(self, msg: Path) -> None:
        self.left_msg = msg
        if self.publish_on_each_update:
            self.publish_scan()

    def cb_right(self, msg: Path) -> None:
        self.right_msg = msg
        if self.publish_on_each_update:
            self.publish_scan()

    def cb_objects(self, msg: Path) -> None:
        self.objects_msg = msg
        if self.publish_on_each_update:
            self.publish_scan()

    def _extract_xyz_from_path(self, path_msg: Optional[Path]) -> List[Tuple[float, float, float]]:
        if path_msg is None:
            return []

        pts: List[Tuple[float, float, float]] = []
        for pose_stamped in path_msg.poses:
            p = pose_stamped.pose.position
            pts.append((float(p.x), float(p.y), float(p.z)))
        return pts

    def _build_scan(self, source_header, pts_xyz: List[Tuple[float, float, float]]) -> LaserScan:
        scan = LaserScan()
        scan.header.stamp = source_header.stamp
        scan.header.frame_id = self.output_frame if self.output_frame else source_header.frame_id

        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_increment
        scan.time_increment = 0.0
        scan.scan_time = 0.0
        scan.range_min = self.range_min
        scan.range_max = self.range_max

        n_bins = int(math.floor((self.angle_max - self.angle_min) / self.angle_increment)) + 1
        fill_value = float("inf") if self.use_inf else (self.range_max + 1.0)
        ranges = [fill_value] * n_bins

        for x, y, _ in pts_xyz:
            # XY frame:
            # x = forward
            # y = lateral
            # ignore points behind robot
            if x <= 0.0:
                continue

            rng = math.sqrt(x * x + y * y)
            if rng < self.range_min or rng > self.range_max:
                continue

            ang = math.atan2(y, x)
            if ang < self.angle_min or ang > self.angle_max:
                continue

            idx = int((ang - self.angle_min) / self.angle_increment)
            if idx < 0 or idx >= n_bins:
                continue

            if self.keep_closest_only:
                if rng < ranges[idx]:
                    ranges[idx] = rng
            else:
                if math.isinf(ranges[idx]) or rng > ranges[idx]:
                    ranges[idx] = rng

        scan.ranges = ranges
        scan.intensities = []
        return scan

    def publish_scan(self) -> None:
        left_pts = self._extract_xyz_from_path(self.left_msg)
        right_pts = self._extract_xyz_from_path(self.right_msg)
        object_pts = self._extract_xyz_from_path(self.objects_msg)

        all_pts = left_pts + right_pts + object_pts

        if not all_pts:
            return

        # Prefer lane headers first, then objects header
        source_header = None
        if self.left_msg is not None:
            source_header = self.left_msg.header
        elif self.right_msg is not None:
            source_header = self.right_msg.header
        elif self.objects_msg is not None:
            source_header = self.objects_msg.header

        if source_header is None:
            return

        scan_msg = self._build_scan(source_header, all_pts)
        self.pub_scan.publish(scan_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LaneToScanNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()