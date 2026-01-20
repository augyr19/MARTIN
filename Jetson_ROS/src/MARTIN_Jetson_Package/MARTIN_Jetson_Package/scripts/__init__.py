"""
MARTIN Jetson Package - Scripts Module

ROS 2 node implementations for inference and visualization.

Modules:
    SYBIL_node: ROS 2 node for SYBIL inference and detection publishing
    camera_node: ROS 2 node for camera visualization with detection overlay

Nodes:
    - SybilNode: Runs SYBIL inference on RealSense frames, publishes 3D positions and object sizes
    - CameraNode: Subscribes to detection data, visualizes on camera feed with markers

Topics:
    Published by SybilNode:
        - /sybil/3d_position (geometry_msgs/Point): 3D coordinates of detections
        - /sybil/object_size (geometry_msgs/Point): Real-world width and height
    
    Subscribed by CameraNode:
        - /sybil/3d_position (geometry_msgs/Point): 3D coordinates to visualize
        - /sybil/object_size (geometry_msgs/Point): Object dimensions to display
    
    Published by CameraNode:
        - /sybil/annotated_frame (sensor_msgs/Image): Frame with detection markers

Example:
    Run nodes from command line::
        
        # Terminal 1: Start SYBIL inference node
        ros2 run MARTIN_Jetson_Package sybil_node \\
            --ros-args \\
            -p model_weights_path:="/path/to/best.pt" \\
            -p inference_rate:=30 \\
            -p confidence_threshold:=0.531
        
        # Terminal 2: Start camera visualization node
        ros2 run MARTIN_Jetson_Package camera_node \\
            --ros-args \\
            -p display_window:=true \\
            -p visualization_rate:=30

Author: ullmannb
Email: ullmannb@tamu.edu
"""

# Note: Do not import node classes here to avoid circular imports
# Nodes should be imported directly when needed

__all__ = [
    'SYBIL_node',
    'camera_node',
]
