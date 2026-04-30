#!/usr/bin/env python3
import math
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

Segment2D = Tuple[Tuple[float, float], Tuple[float, float]]


class BBoxMarkerToScan(Node):
    """Convert 2D bbox outline markers into a synthetic LaserScan."""

    def __init__(self) -> None:
        super().__init__('bbox_marker_to_scan')

        self.declare_parameter('marker_topic', '/bbox_markers')
        self.declare_parameter('scan_topic', '/bbox_fake_scan')
        self.declare_parameter('target_namespace', 'bbox_2d_outline')
        self.declare_parameter('frame_id', '')
        self.declare_parameter('use_marker_header_stamp', False)
        self.declare_parameter('angle_min', -math.pi)
        self.declare_parameter('angle_max', math.pi)
        self.declare_parameter('angle_increment', math.radians(0.5))
        self.declare_parameter('range_min', 0.05)
        self.declare_parameter('range_max', 30.0)
        self.declare_parameter('scan_time', 0.1)
        self.declare_parameter('time_increment', 0.0)
        self.declare_parameter('use_closing_segment', True)
        self.declare_parameter('only_use_line_strip_and_line_list', True)
        self.declare_parameter('publish_inf_for_no_return', True)
        self.declare_parameter('min_points_per_marker', 2)
        self.declare_parameter('ignore_zero_length_segments', True)

        marker_topic = self.get_parameter('marker_topic').value
        scan_topic = self.get_parameter('scan_topic').value

        self.target_namespace = str(self.get_parameter('target_namespace').value)
        self.frame_id_override = str(self.get_parameter('frame_id').value)
        self.use_marker_header_stamp = bool(self.get_parameter('use_marker_header_stamp').value)
        self.angle_min = float(self.get_parameter('angle_min').value)
        self.angle_max = float(self.get_parameter('angle_max').value)
        self.angle_increment = float(self.get_parameter('angle_increment').value)
        self.range_min = float(self.get_parameter('range_min').value)
        self.range_max = float(self.get_parameter('range_max').value)
        self.scan_time = float(self.get_parameter('scan_time').value)
        self.time_increment = float(self.get_parameter('time_increment').value)
        self.use_closing_segment = bool(self.get_parameter('use_closing_segment').value)
        self.only_line_types = bool(self.get_parameter('only_use_line_strip_and_line_list').value)
        self.publish_inf_for_no_return = bool(self.get_parameter('publish_inf_for_no_return').value)
        self.min_points_per_marker = int(self.get_parameter('min_points_per_marker').value)
        self.ignore_zero_length_segments = bool(self.get_parameter('ignore_zero_length_segments').value)

        if self.angle_increment <= 0.0:
            raise ValueError('angle_increment must be > 0')
        if self.angle_max <= self.angle_min:
            raise ValueError('angle_max must be greater than angle_min')
        if self.range_max <= self.range_min:
            raise ValueError('range_max must be greater than range_min')

        # Best Effort sensor-style QoS
        self.scan_pub = self.create_publisher(
            LaserScan,
            scan_topic,
            qos_profile_sensor_data
        )

        self.marker_sub = self.create_subscription(
            MarkerArray,
            marker_topic,
            self.marker_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(
            'bbox_marker_to_scan started | '
            f'marker_topic={marker_topic} scan_topic={scan_topic} '
            f'namespace={self.target_namespace or "<all>"} '
            f'QoS=BEST_EFFORT'
        )

    def marker_callback(self, msg: MarkerArray) -> None:
        segments: List[Segment2D] = []
        frame_id: Optional[str] = self.frame_id_override or None
        stamp = None

        for marker in msg.markers:
            if marker.action in (Marker.DELETE, Marker.DELETEALL):
                continue

            if self.target_namespace and marker.ns != self.target_namespace:
                continue

            if self.only_line_types and marker.type not in (Marker.LINE_STRIP, Marker.LINE_LIST):
                continue

            if frame_id is None and marker.header.frame_id:
                frame_id = marker.header.frame_id

            if stamp is None and self.use_marker_header_stamp:
                stamp = marker.header.stamp

            segments.extend(self.marker_to_segments(marker))

        if frame_id is None:
            frame_id = 'base_link'

        if stamp is None:
            stamp = self.get_clock().now().to_msg()

        ray_count = int(math.floor((self.angle_max - self.angle_min) / self.angle_increment)) + 1
        ranges: List[float] = []
        intensities: List[float] = []

        for i in range(ray_count):
            angle = self.angle_min + i * self.angle_increment
            distance = self.closest_intersection(angle, segments)
            if distance is None:
                ranges.append(math.inf if self.publish_inf_for_no_return else self.range_max)
                intensities.append(0.0)
            else:
                ranges.append(distance)
                intensities.append(1.0)

        scan = LaserScan()
        scan.header.stamp = stamp
        scan.header.frame_id = frame_id
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_increment
        scan.time_increment = self.time_increment
        scan.scan_time = self.scan_time
        scan.range_min = self.range_min
        scan.range_max = self.range_max
        scan.ranges = ranges
        scan.intensities = intensities
        self.scan_pub.publish(scan)

    def marker_to_segments(self, marker: Marker) -> List[Segment2D]:
        pts = marker.points
        if len(pts) < self.min_points_per_marker:
            return []

        segments: List[Segment2D] = []

        if marker.type == Marker.LINE_LIST:
            for i in range(0, len(pts) - 1, 2):
                self._append_segment(segments, pts[i], pts[i + 1])
            return segments

        for i in range(len(pts) - 1):
            self._append_segment(segments, pts[i], pts[i + 1])

        if self.use_closing_segment and len(pts) >= 3 and not self.same_xy(pts[-1], pts[0]):
            self._append_segment(segments, pts[-1], pts[0])

        return segments

    def _append_segment(self, segments: List[Segment2D], p0: Point, p1: Point) -> None:
        if self.ignore_zero_length_segments and self.same_xy(p0, p1):
            return
        segments.append(((p0.x, p0.y), (p1.x, p1.y)))

    @staticmethod
    def same_xy(p0: Point, p1: Point, eps: float = 1e-9) -> bool:
        return abs(p0.x - p1.x) < eps and abs(p0.y - p1.y) < eps

    def closest_intersection(self, angle: float, segments: List[Segment2D]) -> Optional[float]:
        dx = math.cos(angle)
        dy = math.sin(angle)
        best_t: Optional[float] = None

        for (x1, y1), (x2, y2) in segments:
            hit = self.ray_segment_intersection(dx, dy, x1, y1, x2, y2)
            if hit is None:
                continue
            if hit < self.range_min or hit > self.range_max:
                continue
            if best_t is None or hit < best_t:
                best_t = hit

        return best_t

    @staticmethod
    def ray_segment_intersection(
        dx: float,
        dy: float,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        eps: float = 1e-9,
    ) -> Optional[float]:
        sx = x2 - x1
        sy = y2 - y1
        denom = dx * sy - dy * sx

        if abs(denom) < eps:
            return None

        t = (x1 * sy - y1 * sx) / denom
        u = (x1 * dy - y1 * dx) / denom

        if t < 0.0:
            return None
        if u < -eps or u > 1.0 + eps:
            return None

        return t


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BBoxMarkerToScan()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
