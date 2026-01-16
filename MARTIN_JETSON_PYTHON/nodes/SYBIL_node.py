from sensors.RealSense import RealSenseCamera
from models.SYBIL import SybilModel
from utils.depth_ops import (
    depth_at_pixel,
    bbox_to_xyz_xywh,
    bbox_real_world_size
)

import time
import numpy as np

def main():

    # Initialize RealSense
    camera = RealSenseCamera()

    # Load SYBIL model
    model_path = r"C:\Users\brand\Documents\College\2025\MARTIN\SYBIL\runs\final\yolov8m_best_full_retrain\weights\best.pt"
    SYBIL = SybilModel(model_path)

    intrinsics = camera.get_intrinsics()
    depth_scale = camera.get_depth_scale()

    print("SYBIL node running...")

    while True:
        # Get RGB + depth frames
        rgb, depth = camera.get_frames()
        if rgb is None:
            continue

        # Run SYBIL inference
        results = SYBIL.infer(rgb)

        # YOLO returns a list; take the first result
        boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
        boxes_xywh = results[0].boxes.xywh.cpu().numpy()

        for i in range(len(boxes_xyxy)):
            xyxy = boxes_xyxy[i]
            xywh = boxes_xywh[i]

            # 3D coordinate of the object
            xyz = bbox_to_xyz_xywh(xywh, depth, intrinsics, depth_scale)

            # Real-world size of the object
            size = bbox_real_world_size(xyxy, xywh, depth, intrinsics, depth_scale)

            print("\nDetection:")
            print(f"  2D bbox (xyxy): {xyxy}")
            print(f"  3D position (m): {xyz}")
            print(f"  Real-world size (m): {size}")

        time.sleep(0.01)


if __name__ == "__main__":
    main()
