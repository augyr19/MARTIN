import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point

import numpy as np

from martin_jetson_package.sensors.RealSense import RealSenseCamera
from martin_jetson_package.models.SYBIL import SybilModel
from martin_jetson_package.utils.depth_ops import (
    bbox_to_xyz,
    bbox_real_world_size
)


class SybilNode(Node):
    """
    ROS 2 node for running SYBIL inference on RealSense camera input.
    
    Publishes detection results (3D positions and object sizes)
    to topics for downstream processing and visualization.
    
    Topics Published
    ----------------
    - /sybil/3d_position : geometry_msgs/Point (3D coordinate in meters)
    - /sybil/object_size : geometry_msgs/Point (width, height, reserved)
    """

    def __init__(self):
        """Initialize the SYBIL ROS 2 node."""
        super().__init__('sybil_node')
        
        # Declare ROS parameters
        self.declare_parameter('model_weights_path', '')
        self.declare_parameter('inference_rate', 30)  # Hz
        self.declare_parameter('confidence_threshold', 0.531)
        
        # RealSense filter parameters (can be set via launch file or command line)
        self.declare_parameter('enable_spatial_filter', True)
        self.declare_parameter('enable_temporal_filter', True)
        self.declare_parameter('enable_hole_filling', True)
        self.declare_parameter('hole_filling_mode', 3)
        
        # Get parameters
        model_path = self.get_parameter('model_weights_path').value
        self.inference_rate = self.get_parameter('inference_rate').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        
        # Get RealSense filter settings
        enable_spatial = self.get_parameter('enable_spatial_filter').value
        enable_temporal = self.get_parameter('enable_temporal_filter').value
        enable_hole_fill = self.get_parameter('enable_hole_filling').value
        hole_fill_mode = self.get_parameter('hole_filling_mode').value
        
        # Publishers for detection data
        self.detection_pub_xyz = self.create_publisher(
            Point, 
            '/sybil/3d_position', 
            10
        )
        self.detection_pub_size = self.create_publisher(
            Point, 
            '/sybil/object_size', 
            10
        )
        
        # Initialize RealSense with filter parameters
        try:
            self.camera = RealSenseCamera(
                warmup_frames=10,
                apply_spatial_filter=enable_spatial,
                apply_temporal_filter=enable_temporal,
                apply_hole_filling=enable_hole_fill,
                holes_fill_mode=hole_fill_mode
            )
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
        
        # Initialize SYBIL model
        try:
            if not model_path:
                raise ValueError("model_weights_path parameter not set")
            self.sybil = SybilModel(model_path)
            self.get_logger().info(f"SYBIL model loaded from {model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load SYBIL model: {e}")
            raise
        
        # Get camera intrinsics and depth scale
        self.intrinsics = self.camera.get_intrinsics()
        self.depth_scale = self.camera.get_depth_scale()
        
        # Create a timer to run inference at specified rate
        timer_period = 1.0 / self.inference_rate
        self.timer = self.create_timer(timer_period, self.inference_callback)
        
        self.get_logger().info(
            f"SYBIL node started. Publishing at {self.inference_rate} Hz"
        )

    def inference_callback(self):
        """
        Callback for inference timer.
        
        Runs at the specified rate, captures frames, runs SYBIL inference,
        and publishes detection results.
        """
        try:
            # Get RGB + depth frames from RealSense (with filters applied automatically)
            rgb, depth = self.camera.get_frames()
            
            if rgb is None or depth is None:
                self.get_logger().warn("Failed to retrieve frames from camera")
                return
            
            # Run SYBIL inference
            results = self.sybil.infer(rgb, conf_threshold=self.conf_threshold)
            
            if not results or len(results) == 0:
                return
            
            # YOLO returns a list; take the first result
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            boxes_xywh = results[0].boxes.xywh.cpu().numpy()
            
            # Process each detection
            for i in range(len(boxes_xyxy)):
                xyxy = boxes_xyxy[i]
                xywh = boxes_xywh[i]
                
                # Calculate 3D coordinate of the object center
                xyz = bbox_to_xyz(xywh, depth, self.intrinsics, self.depth_scale)
                
                # Calculate real-world size of the object
                size = bbox_real_world_size(
                    xyxy, xywh, depth, self.intrinsics, self.depth_scale
                )
                
                # Log detection to ROS logger
                self.get_logger().info(
                    f"Detection {i}:\n"
                    f"  2D bbox (xyxy): {xyxy}\n"
                    f"  3D position (m): {xyz}\n"
                    f"  Real-world size (m): {size}"
                )
                
                # Publish 3D position as Point message
                if xyz is not None:
                    pos_msg = Point()
                    pos_msg.x = float(xyz[0])
                    pos_msg.y = float(xyz[1])
                    pos_msg.z = float(xyz[2])
                    self.detection_pub_xyz.publish(pos_msg)
                
                # Publish object size as Point message (width, height, reserved)
                if size is not None:
                    size_msg = Point()
                    size_msg.x = float(size[0])  # width
                    size_msg.y = float(size[1])  # height
                    size_msg.z = 0.0  # reserved for future use
                    self.detection_pub_size.publish(size_msg)
        
        except Exception as e:
            self.get_logger().error(f"Error in inference callback: {e}")

    def destroy_node(self):
        """Cleanup when node is shut down."""
        self.get_logger().info("Shutting down SYBIL node...")
        try:
            self.camera.stop()
            self.get_logger().info("RealSense camera stopped")
        except Exception as e:
            self.get_logger().error(f"Error stopping camera: {e}")
        
        super().destroy_node()


def main(args=None):
    """Entry point for the SYBIL node."""
    rclpy.init(args=args)
    
    try:
        sybil_node = SybilNode()
        rclpy.spin(sybil_node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error starting SYBIL node: {e}")
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
