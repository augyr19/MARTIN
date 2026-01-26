import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge


import cv2
import numpy as np
from collections import deque
import threading



class CameraNode(Node):
    """
    ROS 2 node for visualizing SYBIL and AprilTag detections on the camera feed.
    
    Subscribes to:
    - /camera/color/image_raw : Live RGB frames from RealSense
    - /camera/aligned_depth_to_color/image_raw : Aligned depth frames
    - /camera/color/camera_info : Camera intrinsics
    - /sybil/three_d_position : SYBIL detection 3D coordinates
    - /sybil/object_size : SYBIL detection real-world dimensions
    - /apriltag/three_d_position : AprilTag detection 3D coordinates
    
    Publishes:
    - /sybil/annotated_frame : Annotated frame with detection markers
    """


    def __init__(self):
        """Initialize the camera visualization node."""
        super().__init__('camera_node')
        
        # Declare parameters for visualization
        self.declare_parameter('display_window', True)
        self.declare_parameter('visualization_rate', 30)  # Hz
        self.declare_parameter('marker_size', 10)  # Circle radius in pixels
        self.declare_parameter('marker_color', [0, 255, 0])  # BGR: Green
        
        # Get visualization parameters
        self.display_window = self.get_parameter('display_window').value
        viz_rate = self.get_parameter('visualization_rate').value
        self.marker_size = self.get_parameter('marker_size').value
        marker_color = self.get_parameter('marker_color').value
        self.marker_color = tuple(marker_color)
        
        # CV Bridge for image conversion
        self.bridge = CvBridge()
        
        # Storage for camera data (thread-safe)
        self.data_lock = threading.Lock()
        self.rgb = None
        self.depth = None
        self.intrinsics = None
        self.depth_scale = 0.001  # Default RealSense scale (1mm per unit)
        
        # Storage for detection data
        self.latest_positions = deque(maxlen=50)  # Keep last 50 detections
        self.latest_sizes = deque(maxlen=50)
        self.apriltag_positions = deque(maxlen=50)
        
        # Subscribers for camera data
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.rgb_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/aligned_depth_to_color/image_raw',
            self.depth_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            '/camera/color/camera_info',
            self.camera_info_callback,
            10
        )
        
        # Subscribers for SYBIL detection data
        self.pos_sub = self.create_subscription(
            Point,
            '/sybil/three_d_position',
            self.position_callback,
            10
        )
        
        self.size_sub = self.create_subscription(
            Point,
            '/sybil/object_size',
            self.size_callback,
            10
        )
        
        # Subscriber for AprilTag detection data
        self.apriltag_pos_sub = self.create_subscription(
            Point,
            '/apriltag/three_d_position',
            self.apriltag_position_callback,
            10
        )
        
        # Publisher for annotated frame
        self.annotated_pub = self.create_publisher(
            Image,
            '/sybil/annotated_frame',
            10
        )
        
        # Timer for visualization
        timer_period = 1.0 / viz_rate
        self.viz_timer = self.create_timer(timer_period, self.visualization_callback)
        
        self.get_logger().info(
            f"Camera visualization node started. Publishing at {viz_rate} Hz"
        )


    def apriltag_position_callback(self, msg: Point):
        """Store AprilTag 3D positions."""
        with self.data_lock:
            self.apriltag_positions.append({
                'x': msg.x,
                'y': msg.y,
                'z': msg.z
            })
            self.get_logger().debug(
                f"[AprilTag] Received 3D position: ({msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f})"
            )


    def rgb_callback(self, msg: Image):
        """Store latest RGB frame."""
        try:
            with self.data_lock:
                self.rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert RGB message: {e}")


    def depth_callback(self, msg: Image):
        """Store latest depth frame."""
        try:
            with self.data_lock:
                self.depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().warn(f"Failed to convert depth message: {e}")


    def camera_info_callback(self, msg: CameraInfo):
        """Extract intrinsics from camera info (called once)."""
        if self.intrinsics is None:
            # msg.k is a tuple/list: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
            K = msg.k  # lowercase 'k'
            self.intrinsics = {
                'fx': K[0],
                'fy': K[4],
                'ppx': K[2],
                'ppy': K[5],
                'width': msg.width,
                'height': msg.height
            }
            self.get_logger().info(
                f"Camera intrinsics received: fx={self.intrinsics['fx']:.2f}, "
                f"fy={self.intrinsics['fy']:.2f}, "
                f"resolution={self.intrinsics['width']}x{self.intrinsics['height']}"
            )


    def position_callback(self, msg: Point):
        """Callback for SYBIL 3D position detections."""
        with self.data_lock:
            self.latest_positions.append({
                'x': msg.x,
                'y': msg.y,
                'z': msg.z
            })
            self.get_logger().debug(
                f"[SYBIL] Received 3D position: ({msg.x:.2f}, {msg.y:.2f}, {msg.z:.2f})"
            )


    def size_callback(self, msg: Point):
        """Callback for SYBIL object size detections."""
        with self.data_lock:
            self.latest_sizes.append({
                'width': msg.x,
                'height': msg.y
            })


    def visualization_callback(self):
        """
        Callback for visualization timer.
        
        Projects 3D detection coordinates onto the 2D image plane
        and visualizes them on the RGB frame.
        Draws both SYBIL (green) and AprilTag (blue) detections.
        """
        try:
            with self.data_lock:
                if self.rgb is None or self.intrinsics is None:
                    return
                
                # Make a copy to annotate
                annotated_frame = self.rgb.copy()
                
                # Create lists of current detections
                sybil_positions = list(self.latest_positions)
                sybil_sizes = list(self.latest_sizes)
                apriltag_positions = list(self.apriltag_positions)
            
            # Draw SYBIL detections (green)
            num_sybil = min(len(sybil_positions), len(sybil_sizes))
            for i in range(num_sybil):
                pos = sybil_positions[i]
                size = sybil_sizes[i]
                
                pixel_coord = self._project_3d_to_2d(pos)
                
                if pixel_coord is not None:
                    self._draw_detection_marker(
                        annotated_frame, pixel_coord, pos, size, i,
                        label_prefix="SYBIL", color=(0, 255, 0)  # Green
                    )
            
            # Draw AprilTag detections (blue)
            for i, pos in enumerate(apriltag_positions):
                pixel_coord = self._project_3d_to_2d(pos)
                
                if pixel_coord is not None:
                    # No size for AprilTag, so use dummy size
                    dummy_size = {'width': 0, 'height': 0}
                    self._draw_detection_marker(
                        annotated_frame, pixel_coord, pos, dummy_size, i,
                        label_prefix="AprilTag", color=(255, 0, 0)  # Blue
                    )
            
            # Publish annotated frame
            annotated_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding='bgr8')
            self.annotated_pub.publish(annotated_msg)
            
            # Display frame if enabled
            if self.display_window:
                cv2.imshow("SYBIL & AprilTag Detections", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    pass

        except Exception as e:
            self.get_logger().error(f"Error in visualization callback: {e}")



    def _project_3d_to_2d(self, pos_3d: dict) -> tuple:
        """
        Projects a 3D point in camera space to 2D pixel coordinates.
        
        Uses camera intrinsics to map the 3D detection position
        back onto the 2D image plane.
        
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
            if self.intrinsics is None:
                return None
            
            # Extract 3D point
            x, y, z = pos_3d['x'], pos_3d['y'], pos_3d['z']
            
            # Avoid division by zero
            if z <= 0:
                return None
            
            # Project using intrinsics: u = fx * x/z + ppx, v = fy * y/z + ppy
            u = (self.intrinsics['fx'] * x / z) + self.intrinsics['ppx']
            v = (self.intrinsics['fy'] * y / z) + self.intrinsics['ppy']
            
            # Check if pixel is within frame bounds
            if (0 <= u < self.intrinsics['width'] and 
                0 <= v < self.intrinsics['height']):
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
        detection_id: int,
        label_prefix: str = "Detection",
        color: tuple = None):
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
        label_prefix : str
            Prefix for the label (e.g., "SYBIL", "AprilTag")
        color : tuple
            BGR color tuple (use instance marker_color if None)
        """
        if color is None:
            color = self.marker_color
            
        x_pixel, y_pixel = pixel_coord
        
        # Draw a circle at the detection location
        cv2.circle(frame, (x_pixel, y_pixel), self.marker_size, color, -1)
        
        # Draw a crosshair
        crosshair_size = self.marker_size + 5
        cv2.line(frame, (x_pixel - crosshair_size, y_pixel), 
                (x_pixel + crosshair_size, y_pixel), color, 1)
        cv2.line(frame, (x_pixel, y_pixel - crosshair_size), 
                (x_pixel, y_pixel + crosshair_size), color, 1)
        
        # Create annotation text
        text_lines = [
            f"{label_prefix} ID: {detection_id + 1}",
            f"Pos: ({pos_3d['x']:.2f}, {pos_3d['y']:.2f}, {pos_3d['z']:.2f}m)",
        ]
        
        # Only add size if it's non-zero (SYBIL has size, AprilTag doesn't)
        if size['width'] > 0 and size['height'] > 0:
            text_lines.append(f"Size: {size['width']:.2f}x{size['height']:.2f}m")
        
        # Draw text above the marker
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
                color,
                1
            )



    def destroy_node(self):
        """Cleanup when node is shut down."""
        self.get_logger().info("Shutting down camera visualization node...")
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
