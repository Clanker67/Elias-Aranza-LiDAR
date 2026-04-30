#!/home/gcavallo/yolo_env/bin/python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from vision_msgs.msg import Detection2DArray, Detection2D
from vision_msgs.msg import ObjectHypothesisWithPose
from vision_msgs.msg import BoundingBox2D

from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Float32

from ultralytics import YOLO
import numpy as np
import cv2
import threading
import torch
import math


class YoloDetectorNode(Node):
    """
    Publishes:
      - /yolo/detections              (vision_msgs/Detection2DArray)
          bbox + class + score
          NOTE: bbox.center is set to BOTTOM-CENTER pixel (uC, vB)

      - /yolo/annotated              (sensor_msgs/Image) annotated RGB image

      - /yolo/bottom_left_xyz        (geometry_msgs/PointStamped) XYZ from depth at bottom-left pixel (camera frame)
      - /yolo/bottom_center_xyz      (geometry_msgs/PointStamped) XYZ from depth at bottom-center pixel (camera frame)
      - /yolo/bottom_right_xyz       (geometry_msgs/PointStamped) XYZ from depth at bottom-right pixel (camera frame)

      - /yolo/bottom_left_world      (geometry_msgs/PointStamped) latest world BL point in XY plane: (Zc, Xc, 0)
      - /yolo/bottom_center_world    (geometry_msgs/PointStamped) latest world BC point in XY plane: (Zc, Xc, 0)
      - /yolo/bottom_right_world     (geometry_msgs/PointStamped) latest world BR point in XY plane: (Zc, Xc, 0)

      - /yolo/objects_world_path     (nav_msgs/Path) ALL object world points in current frame
                                     stored in XY plane as:
                                     BL, BC, BR for object 1,
                                     BL, BC, BR for object 2, ...

      - /yolo/bbox_width_px          (std_msgs/Float32) bbox width in pixels (x2-x1)
      - /yolo/bbox_width_m           (std_msgs/Float32) width in meters computed in XY plane
                                     using distance between bottom-left & bottom-right WORLD points:
                                     width = hypot(XR-XL, YR-YL)
    """

    def __init__(self):
        super().__init__("yolo_detector_node")

        # -------- Parameters --------
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")

        self.declare_parameter("conf", 0.25)
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("device", "cpu")  # GPU default
        self.declare_parameter("process_hz", 20.0)

        self.declare_parameter(
            "model_path",
            "/home/sd1/ros2_santora_ws/src/camera/models/yolov11_custom.pt",
        )

        # If depth at the exact bottom pixel is noisy/missing, sample slightly above bottom:
        self.declare_parameter("bottom_offset_frac", 0.0)

        # Median patch radius for depth sampling around (u,v)
        self.declare_parameter("depth_patch_r", 2)

        # FP16 inference toggle
        self.declare_parameter("use_fp16", False)

        image_topic = self.get_parameter("image_topic").value
        depth_topic = self.get_parameter("depth_topic").value
        caminfo_topic = self.get_parameter("camera_info_topic").value

        self.conf = float(self.get_parameter("conf").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.device = self.get_parameter("device").value
        self.process_hz = float(self.get_parameter("process_hz").value)

        self.bottom_offset_frac = float(self.get_parameter("bottom_offset_frac").value)
        self.depth_patch_r = int(self.get_parameter("depth_patch_r").value)

        self.use_fp16 = bool(self.get_parameter("use_fp16").value)

        model_path = self.get_parameter("model_path").value

        # -------- Load YOLO model --------
        self.get_logger().info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

        self.bridge = CvBridge()

        # -------- Latest-only buffers --------
        self._lock = threading.Lock()
        self.color_image = None
        self.color_header = None
        self.depth_image = None
        self.K = None
        self.depth_is_mm = True  # RealSense aligned depth often uint16 mm

        # -------- Publishers --------
        self.pub_detections = self.create_publisher(Detection2DArray, "/yolo/detections", 10)
        self.pub_annotated = self.create_publisher(Image, "/yolo/annotated", qos_profile_sensor_data)

        self.pub_bl_xyz = self.create_publisher(PointStamped, "/yolo/bottom_left_xyz", 10)
        self.pub_bc_xyz = self.create_publisher(PointStamped, "/yolo/bottom_center_xyz", 10)
        self.pub_br_xyz = self.create_publisher(PointStamped, "/yolo/bottom_right_xyz", 10)

        # World points in XY plane (Z forced to 0)
        self.pub_bl_world = self.create_publisher(PointStamped, "/yolo/bottom_left_world", 10)
        self.pub_bc_world = self.create_publisher(PointStamped, "/yolo/bottom_center_world", 10)
        self.pub_br_world = self.create_publisher(PointStamped, "/yolo/bottom_right_world", 10)

        # Combined path for ALL objects in current frame
        self.pub_objects_world_path = self.create_publisher(Path, "/yolo/objects_world_path", 10)

        self.pub_width_px = self.create_publisher(Float32, "/yolo/bbox_width_px", 10)
        self.pub_width_m = self.create_publisher(Float32, "/yolo/bbox_width_m", 10)

        # -------- Subscribers --------
        self.sub_image = self.create_subscription(Image, image_topic, self.image_cb, qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(Image, depth_topic, self.depth_cb, qos_profile_sensor_data)
        self.sub_info = self.create_subscription(CameraInfo, caminfo_topic, self.info_cb, qos_profile_sensor_data)

        # -------- Timer loop --------
        period = 1.0 / max(self.process_hz, 1e-3)
        self.timer = self.create_timer(period, self.process_latest)

        self.get_logger().info(f"Color: {image_topic}")
        self.get_logger().info(f"Depth: {depth_topic}")
        self.get_logger().info(f"Info:  {caminfo_topic}")
        self.get_logger().info(f"GPU device: {self.device}")
        self.get_logger().info(f"Processing latest @ {self.process_hz:.1f} Hz")
        self.get_logger().info(f"bottom_offset_frac: {self.bottom_offset_frac:.3f}")
        self.get_logger().info(f"depth_patch_r: {self.depth_patch_r}")
        self.get_logger().info(f"use_fp16: {self.use_fp16}")
        self.get_logger().info("Using transformed world coordinates in XY plane with Z=0")
        self.get_logger().info("Publishing all detected object world points to /yolo/objects_world_path")

    # ---- callbacks (store only) ----
    def info_cb(self, msg: CameraInfo):
        with self._lock:
            self.K = msg.k  # [fx,0,cx, 0,fy,cy, 0,0,1]

    def depth_cb(self, msg: Image):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        with self._lock:
            self.depth_image = depth
            if depth is not None:
                self.depth_is_mm = (depth.dtype == np.uint16)

    def image_cb(self, msg: Image):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        with self._lock:
            self.color_image = img
            self.color_header = msg.header

    # ---- helpers ----
    def pixel_to_xyz(self, u: int, v: int, z_m: float, K):
        fx = K[0]
        fy = K[4]
        cx = K[2]
        cy = K[5]
        X = (u - cx) * z_m / fx
        Y = (v - cy) * z_m / fy
        Z = z_m
        return float(X), float(Y), float(Z)

    def depth_median_patch(self, depth, u: int, v: int, r: int):
        """Median depth around (u,v) in the depth image. Returns float depth value (raw units)."""
        if depth is None:
            return None
        h, w = depth.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return None
        u0 = max(0, u - r)
        u1 = min(w, u + r + 1)
        v0 = max(0, v - r)
        v1 = min(h, v + r + 1)
        patch = depth[v0:v1, u0:u1].astype(np.float32)
        patch = patch[np.isfinite(patch)]
        patch = patch[patch > 0.0]
        if patch.size == 0:
            return None
        return float(np.median(patch))

    def _set_bbox_center(self, bbox: BoundingBox2D, cx: float, cy: float):
        """Robustly set bbox.center across vision_msgs variants."""
        center_t = type(bbox.center)
        c = center_t()
        if hasattr(c, "x"):
            c.x = float(cx)
        if hasattr(c, "y"):
            c.y = float(cy)
        if hasattr(c, "theta"):
            c.theta = 0.0
        bbox.center = c

    def _pub_point(self, pub, header, X, Y, Z):
        p = PointStamped()
        p.header = header
        p.point.x = float(X)
        p.point.y = float(Y)
        p.point.z = float(Z)
        pub.publish(p)

    def _append_point_to_path(self, path_msg: Path, header, X, Y, Z):
        ps = PoseStamped()
        ps.header = header
        ps.pose.position.x = float(X)
        ps.pose.position.y = float(Y)
        ps.pose.position.z = float(Z)
        ps.pose.orientation.w = 1.0
        path_msg.poses.append(ps)

    def _to_meters(self, d_raw: float, depth_is_mm: bool):
        if d_raw is None:
            return None
        return float(d_raw) * 0.001 if depth_is_mm else float(d_raw)

    def process_latest(self):
        with self._lock:
            if self.color_image is None or self.color_header is None:
                return
            color = self.color_image.copy()
            header = self.color_header
            depth = self.depth_image
            K = self.K
            depth_is_mm = self.depth_is_mm

        use_half = bool(self.use_fp16) and torch.cuda.is_available()

        try:
            with torch.no_grad():
                results = self.model.predict(
                    source=color,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                    half=use_half,
                )
        except TypeError:
            with torch.no_grad():
                results = self.model.predict(
                    source=color,
                    conf=self.conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
        except RuntimeError as e:
            self.get_logger().error(f"YOLO predict failed: {e}")
            return

        r0 = results[0]
        annotated = r0.plot()

        det_array = Detection2DArray()
        det_array.header = header

        # Create one combined path for ALL object world points in this frame
        objects_world_path = Path()
        objects_world_path.header = header

        have_depth = (depth is not None) and (K is not None)
        Himg, Wimg = annotated.shape[:2]

        if r0.boxes is not None and len(r0.boxes) > 0:
            boxes = r0.boxes.xyxy.detach().cpu().numpy()
            scores = r0.boxes.conf.detach().cpu().numpy()
            classes = r0.boxes.cls.detach().cpu().numpy().astype(int)

            for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, classes):
                w_px = float(x2 - x1)
                h_px = float(y2 - y1)

                v_float = float(y2) - self.bottom_offset_frac * h_px
                v = int(round(v_float))

                uL = int(round(x1))
                uR = int(round(x2))
                uC = int(round((x1 + x2) / 2.0))

                uL = int(np.clip(uL, 0, Wimg - 1))
                uR = int(np.clip(uR, 0, Wimg - 1))
                uC = int(np.clip(uC, 0, Wimg - 1))
                v = int(np.clip(v, 0, Himg - 1))

                # ---- Detection2D message ----
                det = Detection2D()
                det.bbox = BoundingBox2D()

                self._set_bbox_center(det.bbox, float(uC), float(v))
                det.bbox.size_x = float(w_px)
                det.bbox.size_y = float(h_px)

                hyp = ObjectHypothesisWithPose()
                hyp.hypothesis.class_id = str(cls_id)
                hyp.hypothesis.score = float(score)
                det.results.append(hyp)
                det_array.detections.append(det)

                msg_px = Float32()
                msg_px.data = float(w_px)
                self.pub_width_px.publish(msg_px)

                # ---- XYZ from depth ----
                if have_depth:
                    dL = self.depth_median_patch(depth, uL, v, r=self.depth_patch_r)
                    dR = self.depth_median_patch(depth, uR, v, r=self.depth_patch_r)
                    dC = self.depth_median_patch(depth, uC, v, r=self.depth_patch_r)

                    zL = self._to_meters(dL, depth_is_mm)
                    zR = self._to_meters(dR, depth_is_mm)
                    zC = self._to_meters(dC, depth_is_mm)

                    XYZL = XYZR = XYZC = None

                    if zL is not None and np.isfinite(zL) and zL > 0.0:
                        XYZL = self.pixel_to_xyz(uL, v, zL, K)
                        self._pub_point(self.pub_bl_xyz, header, *XYZL)

                    if zR is not None and np.isfinite(zR) and zR > 0.0:
                        XYZR = self.pixel_to_xyz(uR, v, zR, K)
                        self._pub_point(self.pub_br_xyz, header, *XYZR)

                    if zC is not None and np.isfinite(zC) and zC > 0.0:
                        XYZC = self.pixel_to_xyz(uC, v, zC, K)
                        self._pub_point(self.pub_bc_xyz, header, *XYZC)

                    # Match lane node:
                    # new x = old z
                    # new y = old x
                    # new z = 0
                    def to_world_ground(XYZ):
                        if XYZ is None:
                            return None
                        Xc, Yc, Zc = XYZ
                        return (Zc, Xc, 0.0)

                    wL = to_world_ground(XYZL)
                    wR = to_world_ground(XYZR)
                    wC = to_world_ground(XYZC)

                    # Publish debug single-point topics AND append to combined path
                    if wL is not None:
                        self._pub_point(self.pub_bl_world, header, *wL)
                        self._append_point_to_path(objects_world_path, header, *wL)

                    if wC is not None:
                        self._pub_point(self.pub_bc_world, header, *wC)
                        self._append_point_to_path(objects_world_path, header, *wC)

                    if wR is not None:
                        self._pub_point(self.pub_br_world, header, *wR)
                        self._append_point_to_path(objects_world_path, header, *wR)

                    # Width in meters in XY plane
                    if wL is not None and wR is not None:
                        XL, YL, _ = wL
                        XR, YR, _ = wR

                        width_m = math.hypot(XR - XL, YR - YL)

                        msg_m = Float32()
                        msg_m.data = float(width_m)
                        self.pub_width_m.publish(msg_m)

                        cv2.circle(annotated, (uL, v), 4, (0, 0, 255), -1)
                        cv2.circle(annotated, (uR, v), 4, (0, 255, 0), -1)
                        cv2.circle(annotated, (uC, v), 4, (255, 255, 255), -1)
                        cv2.putText(
                            annotated,
                            f"W={width_m:.2f}m",
                            (int(np.clip(uC + 6, 0, Wimg - 1)), int(np.clip(v - 6, 0, Himg - 1))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

        # Publish all object points from this frame together
        if objects_world_path.poses:
            self.pub_objects_world_path.publish(objects_world_path)

        ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        ann_msg.header = header
        self.pub_annotated.publish(ann_msg)
        self.pub_detections.publish(det_array)


def main():
    rclpy.init()
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
