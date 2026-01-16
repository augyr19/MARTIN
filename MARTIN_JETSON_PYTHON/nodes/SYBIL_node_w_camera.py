from sensors.RealSense import RealSenseCamera
from models.SYBIL import SybilModel
import cv2
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

        # for i in range(len(boxes_xyxy)):
        #     xyxy = boxes_xyxy[i]
        #     xywh = boxes_xywh[i]

        #     # 3D coordinate of the object
        #     xyz = bbox_to_xyz_xywh(xywh, depth, intrinsics, depth_scale)

        #     # Real-world size of the object
        #     size = bbox_real_world_size(xyxy, xywh, depth, intrinsics, depth_scale)

        #     print("\nDetection:")
        #     print(f"  2D bbox (xyxy): {xyxy}")
        #     print(f"  3D position (m): {xyz}")
        #     print(f"  Real-world size (m): {size}")

        for i in range(len(boxes_xyxy)):
            xyxy = boxes_xyxy[i]
            xywh = boxes_xywh[i]

            # Extract YOLO info
            x1, y1, x2, y2 = map(int, xyxy)
            cx, cy, w, h = xywh
            cx, cy = int(cx), int(cy)

            cls_id = int(results[0].boxes.cls[i])
            conf = float(results[0].boxes.conf[i])
            class_name = SYBIL.model.names[cls_id]

            # Depth + 3D position
            depth_m = depth[cy, cx] * depth_scale
            xyz = bbox_to_xyz_xywh(xywh, depth, intrinsics, depth_scale)

            # Draw bounding box
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw center point
            cv2.circle(rgb, (cx, cy), 4, (0, 0, 255), -1)

            # Annotation text
            label = f"{class_name} {conf:.2f} | depth: {depth_m:.2f}m"
            if xyz is not None:
                X, Y, Z = xyz
                label += f" | XYZ: ({X:.2f}, {Y:.2f}, {Z:.2f})"

            # Put text above the box
            cv2.putText(rgb, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("SYBIL Detections", rgb) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

        time.sleep(0.01)


if __name__ == "__main__":
    main()
