import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

import numpy as np
import cv2

from MARTIN_Jetson_Package.sensors.RealSense import RealSenseCamera
from MARTIN_Jetson_Package.utils.depth_ops import (
    bbox_to_xyz,
)

from pupil_apriltags import Detector  # pip install pupil-apriltags


class AprilTagModel:
    """
    Wrapper class for AprilTag detection using pupil_apriltags.
    """

    def __init__(
        self,
        tag_family: str = "tag36h11",
        quad_decimate: float = 1.0,
        quad_sigma: float = 0.0,
        refine_edges: bool = True,
    ):
        self.detector = Detector(
            families=tag_family,
            nthreads=1,
            quad_decimate=quad_decimate,
            quad_sigma=quad_sigma,
            refine_edges=refine_edges,
            decode_sharpening=0.25,
            debug=False,
        )  # [web:12]

    def infer(self, frame: np.ndarray):
        """
        Detect AprilTags in an RGB/BGR frame.

        Returns a list of dicts:
        - id: tag id
        - center: (u, v) pixel coordinates
        - corners: list of 4 (u, v) pixel coordinates (counter‑clockwise)
        """
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        detections_raw = self.detector.detect(gray)  # [web:12]

        detections = []
        for det in detections_raw:
            center = tuple(det.center.tolist())        # (u, v) [web:12][web:16]
            corners = [tuple(c) for c in det.corners.tolist()]  # 4x (u, v) [web:12][web:16]

            detections.append(
                {
                    "id": int(det.tag_id),
                    "center": center,
                    "corners": corners,
                }
            )

        return detections


class AprilTagNode(Node):
    """
    ROS 2 node for detecting AprilTags on RealSense camera input.

    Topics Published
    ----------------
    - /apriltag/pixel_center : geometry_msgs/Point (u, v, 0) pixel coordinates
    - /apriltag/3d_position : geometry_msgs/Point (X, Y, Z) in meters (optional)
    """

    def __init__(self):
        super().__init__("apriltag_node")

        # Parameters
        self.declare_parameter("inference_rate", 30)  # Hz
        self.declare_parameter("tag_family", "tag36h11")

        # RealSense filter parameters (mirror SYBIL node)
        self.declare_parameter("enable_spatial_filter", True)
        self.declare_parameter("enable_temporal_filter", True)
        self.declare_parameter("enable_hole_filling", True)
        self.declare_parameter("hole_filling_mode", 3)

        self.inference_rate = self.get_parameter("inference_rate").value
        tag_family = self.get_parameter("tag_family").value

        # RealSense filter settings
        enable_spatial = self.get_parameter("enable_spatial_filter").value
        enable_temporal = self.get_parameter("enable_temporal_filter").value
        enable_hole_fill = self.get_parameter("enable_hole_filling").value
        hole_fill_mode = self.get_parameter("hole_filling_mode").value

        # Publishers
        self.pixel_center_pub = self.create_publisher(
            Point,
            "/apriltag/pixel_center",
            10,
        )
        self.position_pub = self.create_publisher(
            Point,
            "/apriltag/3d_position",
            10,
        )

        # Initialize RealSense
        try:
            self.camera = RealSenseCamera(
                warmup_frames=10,
                apply_spatial_filter=enable_spatial,
                apply_temporal_filter=enable_temporal,
                apply_hole_filling=enable_hole_fill,
                holes_fill_mode=hole_fill_mode,
            )
            self.get_logger().info("RealSense camera initialized successfully")

            filter_status = self.camera.get_filter_status()
            self.get_logger().info(
                f"RealSense filters: "
                f"Spatial={filter_status['spatial_filter']}, "
                f"Temporal={filter_status['temporal_filter']}, "
                f"HoleFilling={filter_status['hole_filling']}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to initialize RealSense: {e}")
            raise

        # Initialize AprilTag model
        try:
            self.apriltag_model = AprilTagModel(tag_family=tag_family)
            self.get_logger().info(f"AprilTag detector initialized with family {tag_family}")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize AprilTag detector: {e}")
            raise

        # Camera intrinsics and depth scale (same helpers you use for SYBIL)
        self.intrinsics = self.camera.get_intrinsics()   # fx, fy, cx, cy, etc. [web:20][web:25]
        self.depth_scale = self.camera.get_depth_scale()

        # Timer
        timer_period = 1.0 / self.inference_rate
        self.timer = self.create_timer(timer_period, self.inference_callback)

        self.get_logger().info(
            f"AprilTag node started. Publishing at {self.inference_rate} Hz"
        )

    def inference_callback(self):
        """
        Capture frames, run AprilTag detection, and publish results.
        """
        try:
            rgb, depth = self.camera.get_frames()

            if rgb is None or depth is None:
                self.get_logger().warn("Failed to retrieve frames from camera")
                return

            detections = self.apriltag_model.infer(rgb)

            if not detections:
                return

            for i, det in enumerate(detections):
                tag_id = det["id"]
                center_u, center_v = det["center"]
                corners = det["corners"]  # [(u,v), ... 4]

                # Log detection
                self.get_logger().info(
                    f"AprilTag detection {i}: id={tag_id}, "
                    f"center pixel=({center_u:.1f}, {center_v:.1f})"
                )

                # Publish pixel center as Point (u, v, 0)
                pixel_msg = Point()
                pixel_msg.x = float(center_u)
                pixel_msg.y = float(center_v)
                pixel_msg.z = 0.0
                self.pixel_center_pub.publish(pixel_msg)

                # OPTIONAL: approximate 3D position from tag center using your bbox_to_xyz
                # Build a minimal bbox around the corners (xywh in pixel space)
                u_coords = [c[0] for c in corners]
                v_coords = [c[1] for c in corners]
                x_min, x_max = min(u_coords), max(u_coords)
                y_min, y_max = min(v_coords), max(v_coords)
                w = x_max - x_min
                h = y_max - y_min

                bbox_xywh = np.array([center_u, center_v, w, h], dtype=np.float32)

                xyz = bbox_to_xyz(
                    bbox_xywh, depth, self.intrinsics, self.depth_scale
                )

                if xyz is not None:
                    pos_msg = Point()
                    pos_msg.x = float(xyz[0])
                    pos_msg.y = float(xyz[1])
                    pos_msg.z = float(xyz[2])
                    self.position_pub.publish(pos_msg)

        except Exception as e:
            self.get_logger().error(f"Error in AprilTag inference callback: {e}")

    def destroy_node(self):
        self.get_logger().info("Shutting down AprilTag node...")
        try:
            self.camera.stop()
            self.get_logger().info("RealSense camera stopped")
        except Exception as e:
            self.get_logger().error(f"Error stopping camera: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    try:
        node = AprilTagNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error starting AprilTag node: {e}")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
