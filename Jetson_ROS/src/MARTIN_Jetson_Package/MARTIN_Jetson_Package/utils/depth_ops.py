import numpy as np
import pyrealsense2 as rs


def pixel_to_3d(
    pixel_x: int,
    pixel_y: int,
    depth_frame: np.ndarray,
    intrinsics: rs.intrinsics,
    depth_scale: float,
    camera=None  # Optional: can use camera.get_depth_at_pixel() instead
) -> tuple:
    """Convert pixel to 3D"""
    # Option A: Use camera object (recommended)
    if camera is not None:
        depth_m = camera.get_depth_at_pixel(depth_frame, pixel_x, pixel_y)
    else:
        # Option B: Manual calculation
        if pixel_x < 0 or pixel_y < 0 or pixel_y >= depth_frame.shape[0] or pixel_x >= depth_frame.shape[1]:
            return None
        depth_raw = depth_frame[pixel_y, pixel_x]
        if depth_raw == 0:
            return None
        depth_m = float(depth_raw * depth_scale)
    
    if depth_m is None:
        return None
    
    X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [pixel_x, pixel_y], depth_m)
    return (float(X), float(Y), float(Z))


def bbox_to_xyz(
    bbox_xywh: tuple,
    depth_frame: np.ndarray,
    intrinsics: rs.intrinsics,
    depth_scale: float
) -> tuple:
    """
    Converts a YOLO bounding box center to a 3D point in camera space.

    Extracts the center pixel from the bounding box and reprojects it
    to 3D using the depth at that location.

    Parameters
    ----------
    bbox_xywh : tuple
        Bounding box in (cx, cy, w, h) format, where:
        - (cx, cy) is the center in pixel coordinates
        - (w, h) is width and height in pixels
    depth_frame : np.ndarray
        640x480 depth map from RealSense.
    intrinsics : rs.intrinsics
        RealSense camera intrinsic parameters.
    depth_scale : float
        Depth scale factor (meters per depth unit).

    Returns
    -------
    (X, Y, Z) : tuple of floats or None
        3D coordinates in camera space (meters).
        Returns None if depth is invalid at the bbox center.

    Notes
    -----
    - Assumes YOLO output format: (cx, cy, w, h) with floating-point values
    - Uses only the center point of the bounding box for 3D projection
    """
    cx, cy, w, h = bbox_xywh
    cx, cy = int(cx), int(cy)

    # Project center pixel to 3D
    return pixel_to_3d(cx, cy, depth_frame, intrinsics, depth_scale)


def bbox_real_world_size(
    bbox_xyxy: tuple,
    bbox_xywh: tuple,
    depth_frame: np.ndarray,
    intrinsics: rs.intrinsics,
    depth_scale: float
) -> tuple:
    """
    Computes real-world width and height of an object using YOLO bounding boxes.

    Uses depth at the bounding box center to reproject the left/right and
    top/bottom edges into camera space, then computes their distances.

    Parameters
    ----------
    bbox_xyxy : tuple
        (x1, y1, x2, y2) pixel coordinates from YOLO:
        - (x1, y1) is top-left corner
        - (x2, y2) is bottom-right corner
    bbox_xywh : tuple
        (cx, cy, w, h) pixel coordinates from YOLO (center format).
    depth_frame : np.ndarray
        640x480 depth map from RealSense.
    intrinsics : rs.intrinsics
        RealSense camera intrinsic parameters.
    depth_scale : float
        Depth scale factor (meters per depth unit).

    Returns
    -------
    (width_m, height_m) : tuple of floats or None
        Real-world object dimensions in meters.
        Returns None if depth is invalid at the bbox center.

    Notes
    -----
    - Assumes depth is approximately constant across the object
    - Width is computed from left/right edge reprojection at center Y
    - Height is computed from top/bottom edge reprojection at center X
    - Use this for distance estimation and size-based filtering

    Example
    -------
    >>> w_m, h_m = bbox_real_world_size(bbox_xyxy, bbox_xywh, depth, intrinsics, scale)
    >>> if w_m and h_m:
    ...     print(f"Object size: {w_m:.2f}m x {h_m:.2f}m")
    """
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy, w, h = bbox_xywh

    # Convert to ints for depth indexing
    cx, cy = int(cx), int(cy)
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    # Get 3D point at the center of the object
    center_3d = pixel_to_3d(cx, cy, depth_frame, intrinsics, depth_scale)

    if center_3d is None:
        return None

    # Use the Z (depth) value of the center
    depth_m = center_3d[2]

    # Deproject left/right edges to get real-world width
    left_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x1, cy], depth_m)
    right_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x2, cy], depth_m)
    width_m = abs(right_3d[0] - left_3d[0])

    # Deproject top/bottom edges to get real-world height
    top_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, y1], depth_m)
    bottom_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, y2], depth_m)
    height_m = abs(bottom_3d[1] - top_3d[1])

    return (float(width_m), float(height_m))
