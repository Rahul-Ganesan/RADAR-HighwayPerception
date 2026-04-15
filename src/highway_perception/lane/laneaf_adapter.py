import cv2
import numpy as np
from highway_perception.types import LaneOutput


class LaneAFAdapter:
    def __init__(self, weights_path: str = ""):
        self.weights_path = weights_path

    def predict(self, frame: np.ndarray, conf_thresh: float = 0.4) -> LaneOutput:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h // 2 :, :] = edges[h // 2 :, :]
        confidence = float(np.clip(mask.mean() / 255.0 * 2.0, 0.0, 1.0))
        if confidence < conf_thresh:
            mask[:] = 0
        return LaneOutput(mask=mask, lane_confidence=confidence)
