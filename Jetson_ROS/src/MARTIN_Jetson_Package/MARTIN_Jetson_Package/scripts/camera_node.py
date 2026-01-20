import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import pyrealsense2 as rs
from collections import deque
import threading

from martin_jetson_package.sensors.RealSense import RealSenseCamera


class CameraNode(Node):
    """
    ROS 2 node for visualizing SYBIL detections on the camera feed.
    
    Subscribes to:
    - /sybil/3d_position : Detection 3D coordinates
    - /sybil/object_size : Detection real-world dimensions
    
    Publishes:
    - /sybil/annotated_frame : Annotated frame with detection markers
    
    This node integrates RealSense directly to capture frames and project
    3D detection coordinates back onto the 2D image plane for visualization.
    """

    def __init__(self):
        """Initialize the camera visualization node."""
        super().__init__('camera_node')
        
        # Declare parameters for visualization
        self.declare_parameter('display_window', True)
        self.declare_parameter('visualization_rate', 30)  # Hz
        self.declare_parameter('marker_size', 10)  # Circle radius in pixels
        self.declare_parameter('marker_color', [0, 255, 0])  # BGR: Green
        
        # RealSense filter parameters (same as SYBIL_node)
        self.declare_parameter('enable_spatial_filter', True)
        self.declare_parameter('enable_temporal_filter', True)
        self.declare_parameter('enable_hole_filling', True)
        self.declare_parameter('hole_filling_mode', 3)
        
        # Get visualization parameters
        self.display_window = self.get_parameter('display_window').value
        viz_rate = self.get_parameter('visualization_rate').value
        self.marker_size = self.get_parameter('marker_size').value
        marker_color = self.get_parameter('marker_color').value
        self.marker_color = tuple(marker_color)
        
        # Get RealSense filter settings
        enable_spatial = self.get_parameter('enable_spatial_filter').value
        enable_temporal = self.get_parameter('enable_temporal_filter').value
        enable_hole_fill = self.get_parameter('enable_hole_filling').value
        hole_fill_mode = self.get_parameter('hole_filling_mode').value
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Initialize RealSense camera with filter parameters
        try:
            self.camera = RealSenseCamera(
                warmup_frames=10,
                apply_spatial_filter=enable_spatial,
                apply_temporal_filter=enable_temporal,
                apply_hole_filling=enable_hole_fill,
                holes_fill_mode=hole_fill_mode
            )
            self.intrinsics = self.camera.get_intrinsics()
            self.depth_scale = self.camera.get_depth_scale()
            self.get_logger().info("RealSense camera initialized successfully")
            
            # Log filter status
            filter_status = self.camera.get_filter_status()
            self.get_logger().info(
                f"RealSense filters: Spatial={filter_status['spatial_filter']}, "
                f"Temporal={filter_status['temporal_filter']}, "
                f"HoleFilling={filter_status['hole_filling']}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to initialize RealSense: {e}")
            raise
        
        # Storage for latest detection data (thread-safe)
        self.detection_lock = threading.Lock()
        self.latest_positions = deque(maxlen=50)  # Keep last 50 detections
        self.latest_sizes = deque(maxlen=50)
        
        # Subscribers for detection data
        self.pos_sub = self.create_subscription(
            Point,
            '/sybil/3d_position',
            self.position_callback,
            10
        )
        
        self.size_sub = self.create_subscription(
            Point,
            '/sybil/object_size',
            self.size_callback,
            10
        )
        
        # Publisher for annotated frame
        self.annotated_pub = self.create_publisher(
            Image,
            '/sybil/annotated_frame',
            10
        )
        
        # Timer for visualization (runs at fixed rate)
        timer_period = 1.0 / viz_rate
        self.viz_timer = self.create_timer(timer_period, self.visualization_callback)
        
        self.get_logger().info(
            f"Camera visualization node started. Publishing at {viz_rate} Hz"
        )

    def position_callback(self, msg: Point):
        """
        Callback for 3D position detections.
        
        Parameters
        ----------
        msg : geometry_msgs/Point
            3D position in camera space (meters)
        """
        with self.detection_lock:
            self.latest_positions.append({
                'x': msg.x,
                'y': msg.y,
                'z': msg.z
            })

    def size_callback(self, msg: Point):
        """
        Callback for object size detections.
        
        Parameters
        ----------
        msg : geometry_msgs/Point
            Object dimensions (width in x, height in y)
        """
        with self.detection_lock:
            self.latest_sizes.append({
                'width': msg.x,
                'height': msg.y
            })

    def visualization_callback(self):
        """
        Callback for visualization timer.
        
        Captures RGB + depth frames from RealSense, projects 3D detection
        coordinates onto the 2D image plane, and visualizes them.
        """
        try:
            # Capture fresh frames from RealSense (with filters applied automatically)
            rgb, depth = self.camera.get_frames()
            
            if rgb is None or depth is None:
                return
            
            # Make a copy to annotate
            annotated_frame = rgb.copy()
            
            # Get latest detections (thread-safe)
            with self.detection_lock:
                if not self.latest_positions:
                    # If no detections, just publish the raw frame
                    annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
                    self.annotated_pub.publish(annotated_msg)
                    if self.display_window:
                        cv2.imshow("SYBIL Detections", annotated_frame)
                        cv2.waitKey(1)
                    return
                
                # Create a list of positions and sizes for this frame
                positions = list(self.latest_positions)
                sizes = list(self.latest_sizes)
            
            # Process each detection
            num_detections = min(len(positions), len(sizes))
            
            for i in range(num_detections):
                pos = positions[i]
                size = sizes[i]
                
                # Project 3D position to 2D pixel
                pixel_coord = self._project_3d_to_2d(pos)
                
                if pixel_coord is not None:
                    # Draw marker and information on frame
                    self._draw_detection_marker(annotated_frame, pixel_coord, pos, size, i)
            
            # Publish annotated frame
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            self.annotated_pub.publish(annotated_msg)
            
            # Display frame if enabled
            if self.display_window:
                cv2.imshow("SYBIL Detections", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass  # Can implement graceful shutdown

        except Exception as e:
            self.get_logger().error(f"Error in visualization callback: {e}")

    def _project_3d_to_2d(self, pos_3d: dict) -> tuple:
        """
        Projects a 3D point in camera space to 2D pixel coordinates.
        
        Uses RealSense's rs2_project_point_to_pixel function to map the
        3D detection position back onto the 2D image plane.
        
        Parameters
        ----------
        pos_3d : dict
            3D position with keys 'x', 'y', 'z' (meters)
        
        Returns
        -------
        (x_pixel, y_pixel) : tuple or None
            2D pixel coordinates if projection is valid, None otherwise
        """
        try:
            # Extract 3D point
            point_3d = [pos_3d['x'], pos_3d['y'], pos_3d['z']]
            
            # Project to 2D using RealSense intrinsics
            pixel_2d = rs.rs2_project_point_to_pixel(self.intrinsics, point_3d)
            
            # Unpack pixel coordinates
            u, v = pixel_2d[0], pixel_2d[1]
            
            # Check if pixel is within frame bounds
            if 0 <= u < self.intrinsics.width and 0 <= v < self.intrinsics.height:
                return (int(u), int(v))
            else:
                return None
        
        except Exception as e:
            self.get_logger().debug(f"Error projecting 3D to 2D: {e}")
            return None

    def _draw_detection_marker(
        self,
        frame: np.ndarray,
        pixel_coord: tuple,
        pos_3d: dict,
        size: dict,
        detection_id: int
    ):
        """
        Draws a detection marker and annotation on the frame.
        
        Parameters
        ----------
        frame : np.ndarray
            The frame to draw on (BGR)
        pixel_coord : tuple
            (x_pixel, y_pixel) pixel location to mark
        pos_3d : dict
            3D position with 'x', 'y', 'z' (meters)
        size : dict
            Object size with 'width', 'height' (meters)
        detection_id : int
            Index of this detection
        """
        x_pixel, y_pixel = pixel_coord
        
        # Draw a circle at the detection location
        cv2.circle(frame, (x_pixel, y_pixel), self.marker_size, self.marker_color, -1)
        
        # Draw a crosshair
        crosshair_size = self.marker_size + 5
        cv2.line(frame, (x_pixel - crosshair_size, y_pixel), 
                 (x_pixel + crosshair_size, y_pixel), self.marker_color, 1)
        cv2.line(frame, (x_pixel, y_pixel - crosshair_size), 
                 (x_pixel, y_pixel + crosshair_size), self.marker_color, 1)
        
        # Create annotation text
        text_lines = [
            f"ID: {detection_id + 1}",
            f"Pos: ({pos_3d['x']:.2f}, {pos_3d['y']:.2f}, {pos_3d['z']:.2f}m)",
            f"Size: {size['width']:.2f}x{size['height']:.2f}m"
        ]
        
        # Draw text above the marker (with background for readability)
        text_x = x_pixel + self.marker_size + 10
        text_y = y_pixel - (len(text_lines) * 15)
        
        for i, line in enumerate(text_lines):
            y_pos = text_y + (i * 15)
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
            )
            
            # Draw semi-transparent background
            cv2.rectangle(
                frame,
                (text_x - 2, y_pos - text_height - 2),
                (text_x + text_width + 2, y_pos + baseline),
                (0, 0, 0),
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                line,
                (text_x, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1
            )

    def destroy_node(self):
        """Cleanup when node is shut down."""
        self.get_logger().info("Shutting down camera visualization node...")
        try:
            self.camera.stop()
            self.get_logger().info("RealSense camera stopped")
        except Exception as e:
            self.get_logger().error(f"Error stopping camera: {e}")
        
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """Entry point for the camera node."""
    rclpy.init(args=args)
    
    try:
        camera_node = CameraNode()
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error starting camera node: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
