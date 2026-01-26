import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

import numpy as np
import cv2

from MARTIN_Jetson_Package.utils.depth_ops import bbox_to_xyz
from pupil_apriltags import Detector


class AprilTagModel:
    """Wrapper class for AprilTag detection using pupil_apriltags."""

    def __init__(
        self,
        tag_family: str = "tag25h9",
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
        )

    def infer(self, frame: np.ndarray):
        """Detect AprilTags in an RGB/BGR frame."""
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        detections_raw = self.detector.detect(gray)

        detections = []
        for det in detections_raw:
            center = tuple(det.center.tolist())
            corners = [tuple(c) for c in det.corners.tolist()]

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
    Subscribes to /camera/color/image_raw and /camera/aligned_depth_to_color/image_raw.

    Topics Published
    ----------------
    - /apriltag/pixel_center : geometry_msgs/Point (u, v, 0) pixel coordinates
    - /apriltag/three_d_position : geometry_msgs/Point (X, Y, Z) in meters
    """

    def __init__(self):
        super().__init__("apriltag_node")

        # Parameters
        self.declare_parameter("tag_family", "tag25h9")
        tag_family = self.get_parameter("tag_family").value

        # Initialize AprilTag detector
        try:
            self.apriltag_model = AprilTagModel(tag_family=tag_family)
            self.get_logger().info(f"AprilTag detector initialized with family {tag_family}")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize AprilTag detector: {e}")
            raise

        # Publishers
        self.pixel_center_pub = self.create_publisher(Point, "/apriltag/pixel_center", 10)
        self.position_pub = self.create_publisher(Point, "/apriltag/three_d_position", 10)

        # Subscribers
        self.bridge = CvBridge()
        self.rgb = None
        self.depth = None
        self.intrinsics = None
        self.depth_scale = 0.001  # Default RealSense scale (1mm per unit)

        self.rgb_sub = self.create_subscription(
            Image,
            "/camera/color/image_raw",
            self.rgb_callback,
            10,
        )

        self.depth_sub = self.create_subscription(
            Image,
            "/camera/aligned_depth_to_color/image_raw",
            self.depth_callback,
            10,
        )

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/color/camera_info",
            self.camera_info_callback,
            10,
        )

        # Timer for inference (30 Hz by default)
        self.timer = self.create_timer(1.0 / 30.0, self.inference_callback)

        self.get_logger().info("AprilTag node started. Subscribing to camera topics.")

    def rgb_callback(self, msg: Image):
        """Store latest RGB frame."""
        try:
            self.rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().warn(f"Failed to convert RGB message: {e}")

    def depth_callback(self, msg: Image):
        """Store latest depth frame."""
        try:
            self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().warn(f"Failed to convert depth message: {e}")

    def camera_info_callback(self, msg: CameraInfo):
        """Extract intrinsics from camera info (call once)."""
        if self.intrinsics is None:
            # Store intrinsics as a simple dict matching rs.intrinsics structure
            # msg.k is a tuple/list: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            K = msg.k
            self.intrinsics = {
                'fx': K[0],
                'fy': K[4],
                'ppx': K[2],
                'ppy': K[5],
            }
            self.get_logger().info(f"Camera intrinsics received: fx={self.intrinsics['fx']}, fy={self.intrinsics['fy']}")

    def inference_callback(self):
        """Run AprilTag detection on latest frames."""
        if self.rgb is None or self.depth is None or self.intrinsics is None:
            return

        try:
            detections = self.apriltag_model.infer(self.rgb)

            if not detections:
                return

            for i, det in enumerate(detections):
                tag_id = det["id"]
                center_u, center_v = det["center"]
                corners = det["corners"]

                self.get_logger().info(
                    f"AprilTag detection {i}: id={tag_id}, "
                    f"center pixel=({center_u:.1f}, {center_v:.1f})"
                )

                # Publish pixel center
                pixel_msg = Point()
                pixel_msg.x = float(center_u)
                pixel_msg.y = float(center_v)
                pixel_msg.z = 0.0
                self.pixel_center_pub.publish(pixel_msg)

                # Compute 3D position from depth
                u_coords = [c[0] for c in corners]
                v_coords = [c[1] for c in corners]
                x_min, x_max = min(u_coords), max(u_coords)
                y_min, y_max = min(v_coords), max(v_coords)
                w = x_max - x_min
                h = y_max - y_min

                bbox_xywh = np.array([center_u, center_v, w, h], dtype=np.float32)

                # Simple depth lookup at center
                depth_at_center = self.get_depth_at_pixel(int(center_u), int(center_v))
                if depth_at_center is not None:
                    # Convert pixel + depth to 3D using intrinsics
                    X = (center_u - self.intrinsics['ppx']) * depth_at_center / self.intrinsics['fx']
                    Y = (center_v - self.intrinsics['ppy']) * depth_at_center / self.intrinsics['fy']
                    Z = depth_at_center

                    pos_msg = Point()
                    pos_msg.x = float(X)
                    pos_msg.y = float(Y)
                    pos_msg.z = float(Z)
                    self.position_pub.publish(pos_msg)

        except Exception as e:
            self.get_logger().error(f"Error in AprilTag inference callback: {e}")

    def get_depth_at_pixel(self, x: int, y: int) -> float:
        """Get depth in meters at pixel (x, y)."""
        if self.depth is None:
            return None
        if x < 0 or y < 0 or y >= self.depth.shape[0] or x >= self.depth.shape[1]:
            return None

        depth_raw = self.depth[y, x]
        if depth_raw == 0:
            return None

        return float(depth_raw * self.depth_scale)


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
