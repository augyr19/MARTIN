from ultralytics import YOLO
from pathlib import Path
import numpy as np

class SybilModel:
    """
    A wrapper class for the SYBIL computer vision model used in MARTIN.
    Encapsulates model loading and inference logic for modular reuse.
    """

    def __init__(self, weights_path: str):
        """
        Initializes the SYBIL model by loading the YOLOv8 weights.

        Parameters
        ----------
        weights_path : str
            Path to the trained YOLOv8 model weights (.pt or .onnx).

        Why it's useful:
        ----------------
        - Keeps model loading encapsulated and reusable.
        - Allows easy swapping of model weights without changing inference logic.
        - Supports future extension (e.g., loading different YOLO variants).
        """
        resolved_path = Path(weights_path)
        if not resolved_path.exists():
            raise FileNotFoundError(f"Model weights not found at: {resolved_path}")
        self.model = YOLO(resolved_path)

    def infer(self, frame: np.ndarray, conf_threshold: float = 0.531):
        """
        Runs inference on a single RGB frame using the SYBIL model.

        Parameters
        ----------
        frame : np.ndarray
            The RGB image frame to run inference on.
        conf_threshold : float, optional
            Confidence threshold for filtering detections (default is 0.531).

        Returns
        -------
        results : list
            A list of detection results from the YOLO model.

        Why it's useful:
        ----------------
        - Abstracts away YOLO's internal API so nodes only need to call `infer()`.
        - Allows confidence threshold tuning per deployment scenario.
        - Keeps inference logic centralized for easier debugging and benchmarking.
        """
        results = self.model(frame, conf=conf_threshold)
        return results
