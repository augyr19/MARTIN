# sensors/RealSense.py

import pyrealsense2 as rs
import numpy as np
import cv2

class RealSenseCamera:
    """
    A reusable RealSense camera interface for MARTIN.
    Handles pipeline configuration, frame alignment, intrinsics,
    and depth scaling. Provides RGB + depth frames ready for CV models.
    """

    def __init__(self, warmup_frames: int = 10):
        """
        Initializes the RealSense pipeline and prepares the camera.

        Why this is useful:
        -------------------
        - Encapsulates all RealSense SDK setup in one place.
        - Ensures RGB and depth frames are aligned pixel-to-pixel.
        - Extracts intrinsics and depth scale for 3D projection.
        - Warms up the camera so exposure + depth are stable.
        """

        # Create pipeline and config
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable RGB stream at 640x480 (RealSense does NOT support 640x640)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Enable depth stream at 640x480
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start pipeline
        profile = self.pipeline.start(self.config)

        # Depth scale (raw units to meters)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Alignment object (align depth to color)
        self.align = rs.align(rs.stream.color)

        # Get intrinsics for 2D to 3D projection
        color_stream = profile.get_stream(rs.stream.color)
        color_profile = color_stream.as_video_stream_profile()
        self.intrinsics = color_profile.get_intrinsics()

        # Warm-up frames
        for _ in range(warmup_frames):
            self.pipeline.wait_for_frames()

    def get_frames(self):
        """
        Retrieves an aligned RGB frame and depth frame.

        Returns
        -------
        rgb : np.ndarray
            640x480 BGR image from the RealSense color sensor.
        depth : np.ndarray
            640x480 depth map (in raw depth units).

        Why this is useful:
        -------------------
        - Provides synchronized, aligned frames for inference.
        - Keeps RealSense-specific logic out of your CV nodes.
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None

        rgb = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(depth_frame.get_data())

        return rgb, depth

    def get_intrinsics(self):
        """Returns RealSense intrinsics for 2D to 3D projection."""
        return self.intrinsics

    def get_depth_scale(self):
        """Returns the depth scale (meters per depth unit)."""
        return self.depth_scale

    def stop(self):
        """Stops the RealSense pipeline."""
        self.pipeline.stop()
