"""
MARTIN Jetson Package - Sensors Module

Hardware interfaces for sensor data acquisition and preprocessing.

Classes:
    RealSenseCamera: Intel RealSense D435i camera interface with post-processing filters

Features:
    - Automatic RGB and depth frame alignment
    - Spatial filtering for noise reduction
    - Temporal filtering for frame-to-frame smoothing
    - Hole filling for missing depth data
    - Camera intrinsics and depth scale management
    - Configurable warmup and filter parameters

Example:
    Initialize camera with all filters enabled::
        
        from MARTIN_Jetson_Package.sensors import RealSenseCamera
        
        camera = RealSenseCamera(
            warmup_frames=10,
            apply_spatial_filter=True,
            apply_temporal_filter=True,
            apply_hole_filling=True,
            holes_fill_mode=3
        )
        
        rgb, depth = camera.get_frames()
        intrinsics = camera.get_intrinsics()
        depth_scale = camera.get_depth_scale()
        
        # Get depth at specific pixel
        depth_m = camera.get_depth_at_pixel(depth, x=320, y=240)

Author: ullmannb
Email: ullmannb@tamu.edu
"""

from MARTIN_Jetson_Package.sensors.RealSense import RealSenseCamera

__all__ = [
    'RealSenseCamera',
]
