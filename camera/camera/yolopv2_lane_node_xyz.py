#!/home/gcavallo/yolopv2_env/bin/python3
import os
import time
import math
import numpy as np
import cv2
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32


class Yolopv2LaneNode(Node):
    """
    Subscribes:
      - color image
      - aligned depth image
      - camera info

    Publishes:
      - lane mask (mono8)
      - optional overlay
      - left lane boundary as Path (polyline)
      - right lane boundary as Path (polyline)
      - optional average width (meters) computed in XY plane
      - debug overlay that draws left/right polylines on the camera image

    Notes:
      - Uses AMP autocast FP16 on CUDA when use_fp16:=true, while keeping model weights/constants FP32
        (avoids TorchScript bias dtype mismatch).
    """

    def __init__(self):
        super().__init__("yolopv2_lane_node")

        # ---------------- Params ----------------
        self.declare_parameter("warmup", False)
        self.warmup = bool(self.get_parameter("warmup").value)

        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("lane_mask_topic", "/yolopv2/lane_mask")
        self.declare_parameter("overlay_topic", "/yolopv2/lane_overlay")
        self.declare_parameter("publish_overlay", True)

        # Left/Right polyline debug overlay topic
        self.declare_parameter("lr_overlay_topic", "/yolopv2/lane_lr_overlay")
        self.declare_parameter("publish_lr_overlay", True)

        # YOLOPv2 paths/settings
        self.declare_parameter("yolopv2_repo", "/home/sd1/ros2_santora_ws/src/camera")
        self.declare_parameter("weights_path", "/home/sd1/ros2_santora_ws/src/camera/models/lane_only.pt")

        self.declare_parameter("device", "cpu")  # "0" or "cpu"
        self.declare_parameter("img_size", 256)

        # AMP autocast toggle (FP16 compute on CUDA, model stays FP32)
        self.declare_parameter("use_fp16", True)

        # Depth + camera_info
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/aligned_depth_to_color/camera_info")

        # Output as Path (polyline)
        self.declare_parameter("left_path_topic", "/yolopv2/left_lane_path")
        self.declare_parameter("right_path_topic", "/yolopv2/right_lane_path")

        # Optional width output
        self.declare_parameter("width_topic", "/yolopv2/drivable_width_m")
        self.declare_parameter("publish_width", True)

        # Sampling: percent along the lane
        self.declare_parameter("sample_fracs", [0.10, 0.30, 0.50, 0.70, 0.90])
        self.declare_parameter("y_band_half", 3)
        self.declare_parameter("lane_area_min", 600)
        self.declare_parameter("bottom_gate_frac", 0.70)

        # Depth filtering
        self.declare_parameter("depth_scale", 0.001)  # uint16 mm -> meters
        self.declare_parameter("max_depth_m", 30.0)
        self.declare_parameter("depth_patch_r", 2)

        # ---------------- Read params ----------------
        self.image_topic = self.get_parameter("image_topic").value
        self.lane_mask_topic = self.get_parameter("lane_mask_topic").value
        self.overlay_topic = self.get_parameter("overlay_topic").value
        self.publish_overlay = bool(self.get_parameter("publish_overlay").value)

        self.lr_overlay_topic = self.get_parameter("lr_overlay_topic").value
        self.publish_lr_overlay = bool(self.get_parameter("publish_lr_overlay").value)

        repo = self.get_parameter("yolopv2_repo").value
        weights_path = self.get_parameter("weights_path").value
        device_str = self.get_parameter("device").value
        self.img_size = int(self.get_parameter("img_size").value)

        self.use_fp16 = bool(self.get_parameter("use_fp16").value)

        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value

        self.left_path_topic = self.get_parameter("left_path_topic").value
        self.right_path_topic = self.get_parameter("right_path_topic").value

        self.width_topic = self.get_parameter("width_topic").value
        self.publish_width = bool(self.get_parameter("publish_width").value)

        self.sample_fracs = list(self.get_parameter("sample_fracs").value)
        self.y_band_half = int(self.get_parameter("y_band_half").value)
        self.lane_area_min = int(self.get_parameter("lane_area_min").value)
        self.bottom_gate_frac = float(self.get_parameter("bottom_gate_frac").value)

        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.depth_patch_r = int(self.get_parameter("depth_patch_r").value)

        # ---------------- YOLOPv2 imports ----------------
        if repo not in os.sys.path:
            os.sys.path.insert(0, repo)

        try:
            from .utils.utils import select_device, lane_line_mask  # type: ignore
        except Exception:
            from utils.utils import select_device, lane_line_mask  # type: ignore

        self.select_device = select_device
        self.lane_line_mask = lane_line_mask

        # Device
        self.device = self.select_device(device_str)

        # ---------------- Load model ----------------
        self.get_logger().info(f"Loading YOLOPv2 lane-only TorchScript: {weights_path}")
        self.model = torch.jit.load(weights_path, map_location=self.device)
        self.model.eval()

        # Use AMP autocast FP16 on CUDA (keep model FP32 to avoid bias dtype mismatch)
        self.amp_enabled = bool(self.use_fp16 and (self.device.type != "cpu"))
        if self.amp_enabled:
            self.get_logger().info("Using AMP autocast FP16 on CUDA (model stays FP32)")
        else:
            self.get_logger().info("Using FP32")

        # Warmup (optional)
        if self.warmup and self.device.type != "cpu":
            dummy = torch.zeros(1, 3, self.img_size, self.img_size, device=self.device).float()
            with torch.no_grad():
                if self.amp_enabled:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        _ = self.model(dummy)
                else:
                    _ = self.model(dummy)

        # ---------------- ROS I/O ----------------
        self.bridge = CvBridge()

        self.pub_mask = self.create_publisher(Image, self.lane_mask_topic, 10)
        self.pub_overlay = self.create_publisher(Image, self.overlay_topic, 10) if self.publish_overlay else None

        self.pub_lr_overlay = (
            self.create_publisher(Image, self.lr_overlay_topic, 10)
            if self.publish_lr_overlay
            else None
        )

        self.pub_left_path = self.create_publisher(Path, self.left_path_topic, 10)
        self.pub_right_path = self.create_publisher(Path, self.right_path_topic, 10)
        self.pub_width = self.create_publisher(Float32, self.width_topic, 10) if self.publish_width else None

        self.sub_img = self.create_subscription(Image, self.image_topic, self.cb_image, qos_profile_sensor_data)
        self.sub_depth = self.create_subscription(Image, self.depth_topic, self.cb_depth, qos_profile_sensor_data)
        self.sub_info = self.create_subscription(CameraInfo, self.camera_info_topic, self.cb_info, 10)

        # Latest depth + intrinsics
        self._depth_m = None
        self.fx = self.fy = self.cx = self.cy = None

        self.get_logger().info(f"Subscribed RGB: {self.image_topic}")
        self.get_logger().info(f"Subscribed Depth: {self.depth_topic}")
        self.get_logger().info(f"Subscribed Info: {self.camera_info_topic}")
        self.get_logger().info(f"Publishing mask: {self.lane_mask_topic}")
        if self.publish_overlay:
            self.get_logger().info(f"Publishing overlay: {self.overlay_topic}")
        self.get_logger().info(f"Publishing left Path: {self.left_path_topic}")
        self.get_logger().info(f"Publishing right Path: {self.right_path_topic}")
        if self.publish_width:
            self.get_logger().info(f"Publishing width: {self.width_topic}")
        if self.publish_lr_overlay:
            self.get_logger().info(f"Publishing L/R overlay: {self.lr_overlay_topic}")
        self.get_logger().info("Forcing Z=0 in deprojection (XY plane only)")

    # ---------------- Callbacks ----------------
    def cb_depth(self, msg: Image):
        depth_raw = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        if depth_raw is None:
            return
        if depth_raw.dtype == np.uint16:
            self._depth_m = depth_raw.astype(np.float32) * self.depth_scale
        else:
            self._depth_m = depth_raw.astype(np.float32)

    def cb_info(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    # ---------------- Helpers ----------------
    def preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        img = cv2.resize(bgr, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)  # CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        t = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return t.float()

    def _median_depth_patch_m(self, depth_m: np.ndarray, u: int, v: int, r: int):
        H, W = depth_m.shape[:2]
        if not (0 <= u < W and 0 <= v < H):
            return None
        u0, u1 = max(0, u - r), min(W, u + r + 1)
        v0, v1 = max(0, v - r), min(H, v + r + 1)
        patch = depth_m[v0:v1, u0:u1].astype(np.float32)
        patch = patch[np.isfinite(patch)]
        patch = patch[patch > 0.0]
        if patch.size == 0:
            return None
        z = float(np.median(patch))
        if (not math.isfinite(z)) or z <= 0.0 or z > self.max_depth_m:
            return None
        return z

    def _deproject_pixel(self, u: int, v: int, Z: float):
        X_old = (u - self.cx) / self.fx * Z
        return (Z, X_old, 0.0)

    def _build_path(self, header, pts_xyz):
        path = Path()
        path.header = header
        path.poses = []
        for (X, Y, Z) in pts_xyz:
            ps = PoseStamped()
            ps.header = header
            ps.pose.position.x = float(X)
            ps.pose.position.y = float(Y)
            ps.pose.position.z = float(Z)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        return path

    def _pick_two_center_lanes(self, labels, stats, H: int, W: int):
        cx_img = 0.5 * W
        y_gate = int(self.bottom_gate_frac * H)

        candidates = []  # (comp_id, x_med_bottom)
        for comp_id in range(1, stats.shape[0]):  # skip background 0
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area < self.lane_area_min:
                continue
            ys, xs = np.where(labels == comp_id)
            m = ys >= y_gate
            if np.count_nonzero(m) < 50:
                continue
            x_med = float(np.median(xs[m]))
            candidates.append((comp_id, x_med))

        left = [c for c in candidates if c[1] < cx_img]
        right = [c for c in candidates if c[1] > cx_img]

        left_id = max(left, key=lambda t: t[1])[0] if left else None
        right_id = min(right, key=lambda t: t[1])[0] if right else None
        return left_id, right_id

    def _sample_component_by_fracs(self, labels, comp_id: int, fracs, y_band_half: int):
        ys, xs = np.where(labels == comp_id)
        if ys.size == 0:
            return []
        y_min = int(np.min(ys))
        y_max = int(np.max(ys))
        out_uv = []
        for f in fracs:
            y_t = int(round(y_min + float(f) * (y_max - y_min)))
            band = (ys >= (y_t - y_band_half)) & (ys <= (y_t + y_band_half))
            if np.count_nonzero(band) == 0:
                continue
            u = int(round(np.median(xs[band])))
            v = int(round(np.median(ys[band])))
            out_uv.append((u, v))
        return out_uv

    # ---------------- Main image callback ----------------
    def cb_image(self, msg: Image):
        t_start = time.time()

        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if bgr is None:
            return
        H0, W0 = bgr.shape[:2]

        inp = self.preprocess(bgr)

        with torch.no_grad():
            if self.amp_enabled:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    ll = self.model(inp)
            else:
                ll = self.model(inp)

            ll_mask = self.lane_line_mask(ll)

        ll_mask_u8 = ll_mask.astype(np.uint8)
        if ll_mask_u8.shape[:2] != (H0, W0):
            ll_mask_u8 = cv2.resize(ll_mask_u8, (W0, H0), interpolation=cv2.INTER_NEAREST)

        lane_mask = (ll_mask_u8 > 0).astype(np.uint8) * 255

        # Publish mask
        out_msg = self.bridge.cv2_to_imgmsg(lane_mask, encoding="mono8")
        out_msg.header = msg.header
        self.pub_mask.publish(out_msg)

        # Overlay (lane tint)
        if self.publish_overlay and self.pub_overlay is not None:
            overlay = bgr.copy()
            overlay[lane_mask > 0] = (
                overlay[lane_mask > 0] * 0.5 + np.array([0, 255, 255]) * 0.5
            ).astype(np.uint8)
            ov_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            ov_msg.header = msg.header
            self.pub_overlay.publish(ov_msg)

        # Need depth + intrinsics for XYZ
        if self._depth_m is None or self.fx is None:
            return
        depth_m = self._depth_m
        if depth_m.shape[:2] != (H0, W0):
            return

        # Connected components: use 0/1 binary
        binary = (lane_mask > 0).astype(np.uint8)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

        left_id, right_id = self._pick_two_center_lanes(labels, stats, H0, W0)

        left_xyz = []
        right_xyz = []
        left_uv = []
        right_uv = []

        if left_id is not None:
            left_uv = self._sample_component_by_fracs(labels, left_id, self.sample_fracs, self.y_band_half)
            for (u, v) in left_uv[:10]:
                z = self._median_depth_patch_m(depth_m, u, v, r=self.depth_patch_r)
                if z is None:
                    continue
                left_xyz.append(self._deproject_pixel(u, v, z))

        if right_id is not None:
            right_uv = self._sample_component_by_fracs(labels, right_id, self.sample_fracs, self.y_band_half)
            for (u, v) in right_uv[:10]:
                z = self._median_depth_patch_m(depth_m, u, v, r=self.depth_patch_r)
                if z is None:
                    continue
                right_xyz.append(self._deproject_pixel(u, v, z))

        # Publish left/right polyline overlay ON THE IMAGE
        if self.publish_lr_overlay and self.pub_lr_overlay is not None:
            dbg = bgr.copy()

            def draw_uv(img, uv_list, color_bgr, label):
                if not uv_list:
                    return
                uv_sorted = sorted(uv_list, key=lambda p: p[1])
                for (u, v) in uv_sorted:
                    cv2.circle(img, (int(u), int(v)), 5, color_bgr, -1)
                if len(uv_sorted) >= 2:
                    pts = np.array([[int(u), int(v)] for (u, v) in uv_sorted],
                                   dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(img, [pts], False, color_bgr, 3)
                u0, v0 = uv_sorted[-1]
                cv2.putText(img, label, (int(u0) + 8, int(v0) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_bgr, 2, cv2.LINE_AA)

            draw_uv(dbg, left_uv,  (255, 0, 0), "LEFT")
            draw_uv(dbg, right_uv, (0, 0, 255), "RIGHT")

            dbg_msg = self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8")
            dbg_msg.header = msg.header
            self.pub_lr_overlay.publish(dbg_msg)

        header = msg.header
        if left_xyz:
            self.pub_left_path.publish(self._build_path(header, left_xyz))
        if right_xyz:
            self.pub_right_path.publish(self._build_path(header, right_xyz))

        # Width between boundaries (XY plane distance)
        if self.publish_width and self.pub_width is not None and left_xyz and right_xyz:
            n = min(len(left_xyz), len(right_xyz))
            widths = []
            for k in range(n):
                XL, YL, _ = left_xyz[k]
                XR, YR, _ = right_xyz[k]
                widths.append(math.sqrt((XR - XL) ** 2 + (YR - YL) ** 2))
            if widths:
                m = Float32()
                m.data = float(np.mean(widths))
                self.pub_width.publish(m)

        dt_ms = (time.time() - t_start) * 1000.0
        self.get_logger().debug(f"lane inference+xyz {dt_ms:.1f} ms")


def main(args=None):
    rclpy.init(args=args)
    node = Yolopv2LaneNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
