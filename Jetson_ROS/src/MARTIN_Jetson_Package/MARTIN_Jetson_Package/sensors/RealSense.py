import pyrealsense2 as rs
import numpy as np
import cv2


class RealSenseCamera:
    """
    A reusable RealSense camera interface for MARTIN with post-processing filters.

    Handles pipeline configuration, frame alignment, intrinsics, depth scaling,
    and optional depth filtering. Provides clean RGB + depth frames ready for CV models.
    
    Features
    --------
    - Automatic frame alignment (depth to color)
    - Spatial and temporal filtering for noise reduction
    - Hole filling for missing depth data
    - Configurable filter parameters
    - Depth scale and intrinsics management
    """

    def __init__(
        self,
        warmup_frames: int = 10,
        apply_spatial_filter: bool = True,
        apply_temporal_filter: bool = True,
        apply_hole_filling: bool = True,
        holes_fill_mode: int = 2
    ):
        """
        Initializes the RealSense pipeline with post-processing filters.

        Parameters
        ----------
        warmup_frames : int, optional
            Number of frames to discard for camera warm-up (default is 10).
        apply_spatial_filter : bool, optional
            Enable spatial filtering to reduce depth noise (default is True).
        apply_temporal_filter : bool, optional
            Enable temporal filtering for smooth depth over time (default is True).
        apply_hole_filling : bool, optional
            Enable hole filling to interpolate missing depth pixels (default is True).
        holes_fill_mode : int, optional
            Hole filling mode: 0=disabled, 1=fill from left, 2=fill nearest, 3=fill farthest
            (default is 2).

        Raises
        ------
        RuntimeError
            If the RealSense camera is not connected or cannot be initialized.

        Why this is useful:
        -------------------
        - Encapsulates all RealSense SDK setup in one place.
        - Ensures RGB and depth frames are aligned pixel-to-pixel.
        - Applies professional-grade filters for better depth quality.
        - Extracts intrinsics and depth scale for 3D projection.
        - Warms up the camera so exposure + depth are stable.
        """
        try:
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

            # # Initialize post-processing filters
            self.apply_spatial_filter = apply_spatial_filter
            self.apply_temporal_filter = apply_temporal_filter
            self.apply_hole_filling = apply_hole_filling
            
            #debugging purposes
            self.spatial_filter = None
            self.temporal_filter = None
            self.hole_filling_filter = None
            # commented out for node debugging
            # if self.apply_spatial_filter:
            #     self.spatial_filter = rs.spatial_filter()
            #     # Configure spatial filter
            #     # Default options are usually good, but you can tune these:
            #     # - Magnitude: Strength of the filter (1-5, default 2)
            #     # - Smooth alpha: Blending factor (0.25-1, default 0.5)
            #     # - Smooth delta: Max distance to blend (1-50, default 20)
            #     self.spatial_filter.set_option(rs.option.holes_fill, 2)
            #     self.get_logger = lambda: self  # Placeholder for logging
            # else:
            #     self.spatial_filter = None

            # if self.apply_temporal_filter:
            #     self.temporal_filter = rs.temporal_filter()
            #     # Configure temporal filter
            #     # - Smooth alpha: Blending factor (0-1, default 0.4)
            #     # - Smooth delta: Max difference to blend (1-100, default 20)
            #     # Default settings work well for most use cases
            # else:
            #     self.temporal_filter = None

            # if self.apply_hole_filling:
            #     self.hole_filling_filter = rs.hole_filling_filter(holes_fill_mode)
            #     # holes_fill_mode options:
            #     # 0: FILL_FROM_LEFT - Fill holes from left edge
            #     # 1: FARTHEST_FROM_AROUND - Fill with farthest valid neighbor
            #     # 2: NEAREST_FROM_AROUND - Fill with nearest valid neighbor
            # else:
            #     self.hole_filling_filter = None

            # Warm-up frames to stabilize camera exposure and depth
            for _ in range(warmup_frames):
                self.pipeline.wait_for_frames()

        except RuntimeError as e:
            raise RuntimeError(f"Failed to initialize RealSense camera: {e}")

    def get_frames(self):
        """
        Retrieves aligned RGB and depth frames with optional post-processing.

        Returns
        -------
        rgb : np.ndarray or None
            640x480 BGR image from the RealSense color sensor.
            Returns None if frames are unavailable.
        depth : np.ndarray or None
            640x480 depth map (in raw depth units, post-filtered if filters are enabled).
            Returns None if frames are unavailable.

        Notes
        -----
        - Frames are automatically aligned so depth and color pixels correspond
        - Depth data is filtered based on initialization parameters
        - Missing depth pixels may be interpolated via hole filling
        """
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None

            # Convert frames to numpy arrays
            rgb = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())

            # Apply post-processing filters to depth
            # Filters are applied in order: spatial → temporal → hole filling
            # commented out for node debugging
            # if self.apply_spatial_filter and self.spatial_filter is not None:
            #     depth_frame_filtered = rs.frame()
            #     depth_frame_filtered = self.spatial_filter.process(depth_frame)
            #     depth = np.asanyarray(depth_frame_filtered.get_data())

            # if self.apply_temporal_filter and self.temporal_filter is not None:
            #     depth_frame_filtered = rs.frame()
            #     depth_frame_filtered = self.temporal_filter.process(depth_frame)
            #     depth = np.asanyarray(depth_frame_filtered.get_data())

            # if self.apply_hole_filling and self.hole_filling_filter is not None:
            #     depth_frame_filtered = rs.frame()
            #     depth_frame_filtered = self.hole_filling_filter.process(depth_frame)
            #     depth = np.asanyarray(depth_frame_filtered.get_data())

            return rgb, depth

        except RuntimeError as e:
            print(f"Error retrieving frames: {e}")
            return None, None

    def get_depth_at_pixel(self, depth_frame: np.ndarray, x: int, y: int) -> float:
        """
        Retrieves the depth (in meters) at a specific pixel with bounds checking.

        Parameters
        ----------
        depth_frame : np.ndarray
            640x480 depth map from RealSense (raw depth units).
        x : int
            Pixel x-coordinate (column).
        y : int
            Pixel y-coordinate (row).

        Returns
        -------
        depth_m : float or None
            Depth in meters at the specified pixel.
            Returns None if pixel is out of bounds or has no depth data (depth_raw == 0).

        Notes
        -----
        - depth_frame is indexed as [row, column] = [y, x]
        - A raw depth value of 0 indicates missing/invalid data
        - If hole filling is enabled, this should rarely return None
        """
        # Bounds checking
        if x < 0 or y < 0 or y >= depth_frame.shape[0] or x >= depth_frame.shape[1]:
            return None

        # depth_frame is indexed as [row, column] = [y, x]
        depth_raw = depth_frame[y, x]

        # Raw depth of 0 means no depth data at this pixel
        if depth_raw == 0:
            return None

        # Convert raw depth units to meters using depth scale
        return float(depth_raw * self.depth_scale)

    def get_intrinsics(self):
        """
        Returns RealSense intrinsics for 3D projection calculations.

        Returns
        -------
        intrinsics : rs.intrinsics
            Camera intrinsic parameters including focal length, principal point,
            and distortion coefficients. Used by rs2_deproject_pixel_to_point and
            rs2_project_point_to_pixel functions.
        """
        return self.intrinsics

    def get_depth_scale(self):
        """
        Returns the depth scale (meters per depth unit).

        Returns
        -------
        depth_scale : float
            Scale factor to convert raw depth units to meters.
            Typically around 0.001 meters (1mm) per unit.
        """
        return self.depth_scale

    def get_filter_status(self) -> dict:
        """
        Returns the current status of all post-processing filters.

        Returns
        -------
        status : dict
            Dictionary with boolean values for each filter:
            - 'spatial_filter': Whether spatial filtering is enabled
            - 'temporal_filter': Whether temporal filtering is enabled
            - 'hole_filling': Whether hole filling is enabled

        Notes
        -----
        Useful for debugging and logging filter configuration.
        """
        return {
            'spatial_filter': self.apply_spatial_filter,
            'temporal_filter': self.apply_temporal_filter,
            'hole_filling': self.apply_hole_filling
        }

    def stop(self):
        """
        Safely stops the RealSense pipeline and releases hardware resources.

        Why this is useful:
        -------------------
        - Ensures proper cleanup and hardware release.
        - Should be called in node shutdown handlers.
        - Prevents resource leaks and connection issues on restart.
        """
        try:
            self.pipeline.stop()
        except Exception as e:
            print(f"Error stopping RealSense pipeline: {e}")
