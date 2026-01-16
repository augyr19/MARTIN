import numpy as np
import pyrealsense2 as rs


def depth_at_pixel(depth_frame, x, y, depth_scale):
    """
    Retrieves the depth (in meters) at a specific pixel.
    """
    if x < 0 or y < 0 or y >= depth_frame.shape[0] or x >= depth_frame.shape[1]:
        return None

    # depth_frame[row, column] leads to depth_frame[y, x]
    depth_raw = depth_frame[y, x]
    if depth_raw == 0:
        return None  # No depth data at this pixel

    return depth_raw * depth_scale


def bbox_to_xyz_xywh(bbox_xywh, depth_frame, intrinsics, depth_scale):
    """
    Converts a YOLO bounding box (xywh format) into a 3D (x, y, z) coordinate.

    Parameters
    ----------
    bbox_xywh : tuple
        Bounding box in (cx, cy, w, h) format.
    depth_frame : np.ndarray
        640x480 depth map.
    intrinsics : rs.intrinsics
        RealSense camera intrinsics.
    depth_scale : float
        Depth scale (meters per unit).

    Returns
    -------
    (X, Y, Z) : tuple of floats
        3D coordinates in camera space (meters).
        Returns None if depth is invalid.
    """
    cx, cy, w, h = bbox_xywh
    cx, cy = int(cx), int(cy)

    depth_m = depth_at_pixel(depth_frame, cx, cy, depth_scale)
    if depth_m is None:
        return None

    X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth_m)
    return X, Y, Z


def bbox_real_world_size(bbox_xyxy, bbox_xywh, depth_frame, intrinsics, depth_scale):
    """
    Computes real-world width and height of an object using YOLO bounding boxes.

    Parameters
    ----------
    bbox_xyxy : tuple
        (x1, y1, x2, y2) pixel coordinates from YOLO.
    bbox_xywh : tuple
        (cx, cy, w, h) pixel coordinates from YOLO.
    depth_frame : np.ndarray
        640x480 depth map.
    intrinsics : rs.intrinsics
        RealSense intrinsics.
    depth_scale : float
        Depth scale in meters.

    Returns
    -------
    (width_m, height_m) : tuple of floats
        Real-world object dimensions in meters.
    """

    x1, y1, x2, y2 = bbox_xyxy
    cx, cy, w, h = bbox_xywh

    # Convert to ints for depth indexing
    cx, cy = int(cx), int(cy)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Depth at the center of the object
    depth_m = depth_at_pixel(depth_frame, cx, cy, depth_scale)
    if depth_m is None:
        return None

    # Deproject left/right edges → width
    left_3d  = rs.rs2_deproject_pixel_to_point(intrinsics, [x1, cy], depth_m)
    right_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x2, cy], depth_m)
    width_m = abs(right_3d[0] - left_3d[0])

    # Deproject top/bottom edges → height
    top_3d    = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, y1], depth_m)
    bottom_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, y2], depth_m)
    height_m = abs(bottom_3d[1] - top_3d[1])

    return width_m, height_m
