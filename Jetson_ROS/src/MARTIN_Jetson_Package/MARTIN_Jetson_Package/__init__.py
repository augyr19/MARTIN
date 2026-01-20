"""
MARTIN Jetson Package - ROS 2 Humble Integration

Multi-Agent Robotic Trash INspection (MARTIN) package for computer vision-based
trash detection using SYBIL YOLO model and Intel RealSense depth cameras.

Modules:
    sensors: Hardware interfaces (RealSense camera)
    models: Computer vision models (SYBIL/YOLOv8 wrapper)
    utils: Utility functions (depth operations, coordinate transforms)
    scripts: ROS 2 node implementations

Key Classes:
    - RealSenseCamera: RealSense D415/D435 camera interface with filters
    - SybilModel: YOLOv8 inference wrapper
    - SybilNode: ROS 2 inference node
    - CameraNode: ROS 2 visualization node

Example:
    Initialize RealSense camera with filters::
        
        from MARTIN_Jetson_Package.sensors.RealSense import RealSenseCamera
        camera = RealSenseCamera(
            apply_spatial_filter=True,
            apply_temporal_filter=True,
            apply_hole_filling=True
        )
        rgb, depth = camera.get_frames()
    
    Load SYBIL model and run inference::
        
        from MARTIN_Jetson_Package.models.SYBIL import SybilModel
        model = SybilModel('/path/to/weights/best.pt')
        results = model.infer(rgb, conf_threshold=0.531)
    
    Project 3D detection to 2D pixel::
        
        from MARTIN_Jetson_Package.utils.depth_ops import bbox_to_xyz
        xyz = bbox_to_xyz(bbox_xywh, depth, intrinsics, depth_scale)

Author: ullmannb
License: MIT
Version: 0.1.0
"""

__version__ = '0.1.0'
__author__ = 'ullmannb'
__email__ = 'ullmannb@tamu.edu'
__license__ = 'MIT'

# Lazy imports - only load modules when needed
def __getattr__(name):
    """Provide lazy loading of submodules."""
    submodules = {
        'sensors': 'MARTIN_Jetson_Package.sensors',
        'models': 'MARTIN_Jetson_Package.models',
        'utils': 'MARTIN_Jetson_Package.utils',
        'scripts': 'MARTIN_Jetson_Package.scripts',
    }
    
    if name in submodules:
        import importlib
        return importlib.import_module(submodules[name])
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'sensors',
    'models',
    'utils',
    'scripts',
    '__version__',
    '__author__',
]
