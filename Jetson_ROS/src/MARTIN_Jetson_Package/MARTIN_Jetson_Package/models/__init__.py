"""
MARTIN Jetson Package - Models Module

Computer vision model wrappers for object detection and inference.

Classes:
    SybilModel: Single-class YOLOv8 Based Identifier of Litter

Features:
    - Encapsulated YOLOv8 model loading and initialization
    - Configurable confidence threshold for detections
    - Simplified inference API for RGB frames
    - Support for .pt and .onnx weight formats
    - Automatic model weight validation

Example:
    Load SYBIL model and run inference::
        
        from MARTIN_Jetson_Package.models import SybilModel
        
        # Initialize model with custom weights
        model = SybilModel('/path/to/weights/best.pt')
        
        # Run inference on RGB frame
        results = model.infer(rgb_frame, conf_threshold=0.531)
        
        # Extract detection results
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        boxes_xywh = results[0].boxes.xywh.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy()

Author: ullmannb
Email: ullmannb@tamu.edu
"""

from MARTIN_Jetson_Package.models.SYBIL import SybilModel

__all__ = [
    'SybilModel',
]
