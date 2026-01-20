"""
MARTIN Jetson Package - Utils Module

Utility functions for depth processing and coordinate transformations.

Functions:
    pixel_to_3d: Convert 2D pixel coordinates to 3D camera space
    bbox_to_xyz: Convert YOLO bounding box center to 3D coordinates
    bbox_real_world_size: Compute real-world dimensions of detected objects

Features:
    - RealSense intrinsics-based 3D projection
    - Automatic depth validation and bounds checking
    - Support for YOLO xywh and xyxy bounding box formats
    - Real-world size estimation from depth maps
    - Robust error handling for invalid depth values

Example:
    Convert detection to 3D coordinates::
        
        from MARTIN_Jetson_Package.utils import bbox_to_xyz, bbox_real_world_size
        
        # Get 3D position of detected object
        xyz = bbox_to_xyz(
            bbox_xywh=(320, 240, 100, 150),
            depth_frame=depth,
            intrinsics=camera_intrinsics,
            depth_scale=camera_depth_scale
        )
        
        # Get real-world size of detected object
        size = bbox_real_world_size(
            bbox_xyxy=(270, 165, 370, 315),
            bbox_xywh=(320, 240, 100, 150),
            depth_frame=depth,
            intrinsics=camera_intrinsics,
            depth_scale=camera_depth_scale
        )
        
        if xyz and size:
            print(f"Position: {xyz} m")
            print(f"Size: {size[0]:.2f} x {size[1]:.2f} m")

Author: ullmannb
Email: ullmannb@tamu.edu
"""

from MARTIN_Jetson_Package.utils.depth_ops import (
    pixel_to_3d,
    bbox_to_xyz,
    bbox_real_world_size,
)

__all__ = [
    'pixel_to_3d',
    'bbox_to_xyz',
    'bbox_real_world_size',
]
