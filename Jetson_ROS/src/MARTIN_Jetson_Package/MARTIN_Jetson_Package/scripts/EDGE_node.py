#!/usr/bin/env python3
import math
from typing import List, Tuple, Optional

import rclpy
from rclpy.node import Node

import numpy as np
import cv2

from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseArray, Pose


def _nan_pose() -> Pose:
    p = Pose()
    p.position.x = float("nan")
    p.position.y = float("nan")
    p.position.z = float("nan")
    p.orientation.w = 1.0
    return p


class EdgeNode(Node):
    """
    EDGE node:
      Subscribes:
        /camera/color/image_raw
        /camera/aligned_depth_to_color/image_raw
        /camera/color/camera_info

      Publishes:
        /edge/lines_3d    geometry_msgs/PoseArray
          - poses are 3D points (x,y,z) in camera frame
          - NaN pose separates different polylines (disconnected edges)
        /edge/debug_image sensor_msgs/Image (optional)
    """

    def __init__(self):
        super().__init__("edge_node")

        # ===== Params =====
        self.declare_parameter("publish_hz", 15.0)

        # Only analyze lower portion of the image (edge near robot)
        self.declare_parameter("roi_y_start_frac", 0.50)

        # Depth handling
        self.declare_parameter("depth_scale", 0.001)   # if depth is uint16 mm
        self.declare_parameter("depth_window", 5)      # odd; median filter window
        self.declare_parameter("min_depth_m", 0.20)
        self.declare_parameter("max_depth_m", 12.0)

        # Morphology + contour cleanup
        self.declare_parameter("morph_kernel", 5)      # odd
        self.declare_parameter("min_contour_len", 80)  # pixels; rejects tiny junk
        self.declare_parameter("max_points_per_line", 200)
        self.declare_parameter("contour_downsample", 6)

        # HSV thresholds (starter values tuned for sunny Texas)
        # Work area = green grass OR tan/dead grass
        self.declare_parameter("green_h_min", 28)
        self.declare_parameter("green_h_max", 95)
        self.declare_parameter("green_s_min", 45)
        self.declare_parameter("green_v_min", 35)

        self.declare_parameter("tan_h_min", 8)
        self.declare_parameter("tan_h_max", 35)
        self.declare_parameter("tan_s_min", 35)
        self.declare_parameter("tan_v_min", 60)

        # Road = gray-ish: low saturation, medium/high value
        self.declare_parameter("road_s_max", 55)
        self.declare_parameter("road_v_min", 40)
        self.declare_parameter("road_v_max", 235)

        self.declare_parameter("publish_debug_image", True)

        # ===== State =====
        self.bridge = CvBridge()
        self.rgb = None
        self.depth = None
        self.intrinsics = None
        self.frame_id = "camera_color_optical_frame"

        # ===== ROS IO =====
        self.lines_pub = self.create_publisher(PoseArray, "/edge/lines_3d", 10)
        self.debug_pub = self.create_publisher(Image, "/edge/debug_image", 10)

        self.rgb_sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.rgb_callback, 10
        )
        self.depth_sub = self.create_subscription(
            Image, "/camera/aligned_depth_to_color/image_raw", self.depth_callback, 10
        )
        self.info_sub = self.create_subscription(
            CameraInfo, "/camera/color/camera_info", self.camera_info_callback, 10
        )

        hz = float(self.get_parameter("publish_hz").value)
        self.timer = self.create_timer(1.0 / hz, self.tick)

        self.get_logger().info("EDGE node started. Publishing /edge/lines_3d as PoseArray.")

    # ---------- Callbacks ----------
    def rgb_callback(self, msg: Image):
        try:
            self.rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"RGB conversion failed: {e}")

    def depth_callback(self, msg: Image):
        try:
            self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")

    def camera_info_callback(self, msg: CameraInfo):
        if self.intrinsics is None:
            K = msg.k
            self.intrinsics = {
                "fx": float(K[0]),
                "fy": float(K[4]),
                "cx": float(K[2]),
                "cy": float(K[5]),
                "width": int(msg.width),
                "height": int(msg.height),
            }
            if msg.header.frame_id:
                self.frame_id = msg.header.frame_id
            self.get_logger().info(
                f"Intrinsics: fx={self.intrinsics['fx']:.2f}, fy={self.intrinsics['fy']:.2f}, "
                f"res={self.intrinsics['width']}x{self.intrinsics['height']}"
            )

    # ---------- Main loop ----------
    def tick(self):
        if self.rgb is None or self.depth is None or self.intrinsics is None:
            return

        bgr = self.rgb
        depth = self.depth

        roi_y0 = int(self.get_parameter("roi_y_start_frac").value * bgr.shape[0])
        roi_y0 = int(np.clip(roi_y0, 0, bgr.shape[0] - 1))

        boundary_mask = self.make_boundary_mask(bgr, roi_y0)
        polylines_uv = self.boundary_to_polylines(boundary_mask, roi_y0)

        if not polylines_uv:
            return

        # Convert all polylines to 3D and publish in one PoseArray with NaN separators
        msg = PoseArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        total_added = 0
        for line_uv in polylines_uv:
            pts_xyz = self.deproject_polyline(line_uv, depth)
            if pts_xyz is None or pts_xyz.shape[0] < 2:
                continue

            # Add line points
            for (x, y, z) in pts_xyz:
                p = Pose()
                p.position.x = float(x)
                p.position.y = float(y)
                p.position.z = float(z)
                p.orientation.w = 1.0
                msg.poses.append(p)
                total_added += 1

            # Separator between disconnected lines
            msg.poses.append(_nan_pose())

        if total_added >= 2:
            self.lines_pub.publish(msg)

        if bool(self.get_parameter("publish_debug_image").value):
            self.publish_debug(bgr, polylines_uv)

    # ---------- Segmentation / boundary ----------
    def make_boundary_mask(self, bgr: np.ndarray, roi_y0: int) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)

        # Green mask
        gh0 = int(self.get_parameter("green_h_min").value)
        gh1 = int(self.get_parameter("green_h_max").value)
        gs0 = int(self.get_parameter("green_s_min").value)
        gv0 = int(self.get_parameter("green_v_min").value)
        green = cv2.inRange(hsv, (gh0, gs0, gv0), (gh1, 255, 255))

        # Tan / dead grass
        th0 = int(self.get_parameter("tan_h_min").value)
        th1 = int(self.get_parameter("tan_h_max").value)
        ts0 = int(self.get_parameter("tan_s_min").value)
        tv0 = int(self.get_parameter("tan_v_min").value)
        tan = cv2.inRange(hsv, (th0, ts0, tv0), (th1, 255, 255))

        work = cv2.bitwise_or(green, tan)

        # Road gray: low saturation, mid-high value
        rs_max = int(self.get_parameter("road_s_max").value)
        rv_min = int(self.get_parameter("road_v_min").value)
        rv_max = int(self.get_parameter("road_v_max").value)
        road = ((S <= rs_max) & (V >= rv_min) & (V <= rv_max)).astype(np.uint8) * 255

        # Morph cleanup
        k = int(self.get_parameter("morph_kernel").value)
        k = k if k % 2 == 1 else k + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, kernel, iterations=1)
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel, iterations=1)

        road = cv2.morphologyEx(road, cv2.MORPH_OPEN, kernel, iterations=1)
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Boundary band: pixels where they touch
        work_d = cv2.dilate(work, kernel, iterations=1)
        road_d = cv2.dilate(road, kernel, iterations=1)
        boundary = cv2.bitwise_or(cv2.bitwise_and(work_d, road), cv2.bitwise_and(road_d, work))

        # Keep only ROI
        boundary[:roi_y0, :] = 0
        return boundary

    def boundary_to_polylines(self, boundary_mask: np.ndarray, roi_y0: int) -> List[List[Tuple[int, int]]]:
        """
        Find disconnected boundary pieces as separate polylines via contours.
        """
        min_len = int(self.get_parameter("min_contour_len").value)
        ds = int(self.get_parameter("contour_downsample").value)
        max_pts = int(self.get_parameter("max_points_per_line").value)

        # Find contours on the boundary band
        contours, _ = cv2.findContours(boundary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        polylines = []
        for c in contours:
            if c.shape[0] < min_len:
                continue

            pts = c[:, 0, :]  # (N,2) as (x,y)

            # Downsample
            pts = pts[::ds].astype(np.int32)
            if pts.shape[0] < 2:
                continue

            # Sort by x for a stable "left-to-right" polyline
            # (Contours are cyclic; this gives a consistent drawing order)
            order = np.argsort(pts[:, 0])
            pts = pts[order]

            # Limit points
            if pts.shape[0] > max_pts:
                idx = np.linspace(0, pts.shape[0] - 1, max_pts).astype(np.int32)
                pts = pts[idx]

            line_uv = [(int(x), int(y)) for x, y in pts]
            polylines.append(line_uv)

        # Prefer longer lines first
        polylines.sort(key=lambda ln: len(ln), reverse=True)
        return polylines

    # ---------- Depth / deprojection ----------
    def depth_m_at(self, depth: np.ndarray, u: int, v: int) -> Optional[float]:
        h, w = depth.shape[:2]
        if u < 0 or v < 0 or u >= w or v >= h:
            return None

        win = int(self.get_parameter("depth_window").value)
        win = win if win % 2 == 1 else win + 1
        r = win // 2

        u0, u1 = max(0, u - r), min(w, u + r + 1)
        v0, v1 = max(0, v - r), min(h, v + r + 1)

        patch = depth[v0:v1, u0:u1].reshape(-1)

        if patch.dtype in (np.uint16, np.uint32):
            scale = float(self.get_parameter("depth_scale").value)
            vals = patch.astype(np.float32) * scale
        else:
            vals = patch.astype(np.float32)

        vals = vals[np.isfinite(vals)]
        vals = vals[vals > 0.0]
        if vals.size == 0:
            return None

        z = float(np.median(vals))

        if z < float(self.get_parameter("min_depth_m").value):
            return None
        if z > float(self.get_parameter("max_depth_m").value):
            return None
        return z

    def deproject_polyline(self, pixels_uv: List[Tuple[int, int]], depth: np.ndarray) -> Optional[np.ndarray]:
        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]

        pts_xyz = []
        for (u, v) in pixels_uv:
            z = self.depth_m_at(depth, u, v)
            if z is None:
                continue

            x = (float(u) - cx) * z / fx
            y = (float(v) - cy) * z / fy
            pts_xyz.append((x, y, z))

        if len(pts_xyz) < 2:
            return None
        return np.array(pts_xyz, dtype=np.float32)

    # ---------- Debug ----------
    def publish_debug(self, bgr: np.ndarray, polylines_uv: List[List[Tuple[int, int]]]):
        dbg = bgr.copy()

        colors = [(0, 0, 255), (0, 255, 255), (255, 0, 0), (255, 0, 255)]
        for i, line in enumerate(polylines_uv[:8]):
            if len(line) < 2:
                continue
            pts = np.array(line, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(dbg, [pts], False, colors[i % len(colors)], 2)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(dbg, encoding="bgr8"))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = EdgeNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
