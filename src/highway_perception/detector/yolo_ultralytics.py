from typing import Dict, List, Optional
import numpy as np
from highway_perception.types import Detection

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class YOLOUltralyticsDetector:
    def __init__(self, model_path: str, class_names: List[str], conf: float = 0.25, iou: float = 0.5):
        self.class_names = class_names
        self.conf = conf
        self.iou = iou
        self.model = YOLO(model_path) if YOLO else None
        # Map source model class names (e.g., COCO) to project labels.
        self.class_name_map: Dict[str, str] = {
            "car": "car",
            "truck": "truck",
            "bus": "bus",
            "motorcycle": "motor",
            "person": "person",
            # BDD "rider" does not have a direct COCO single-class equivalent.
        }

    def _resolve_source_class_name(self, cls_id: int) -> Optional[str]:
        if self.model is None:
            return None
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            return names.get(cls_id)
        if isinstance(names, (list, tuple)) and 0 <= cls_id < len(names):
            return names[cls_id]
        return None

    def predict(self, frame: np.ndarray) -> List[Detection]:
        if self.model is None:
            return []
        results = self.model.predict(frame, conf=self.conf, iou=self.iou, verbose=False)
        if not results:
            return []
        pred = []
        boxes = results[0].boxes
        if boxes is None:
            return pred
        for b in boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            cls_id = int(b.cls[0].item())
            score = float(b.conf[0].item())
            source_name = self._resolve_source_class_name(cls_id)
            mapped_name = self.class_name_map.get(source_name, "")
            if mapped_name not in self.class_names:
                continue
            pred.append(Detection((x1, y1, x2, y2), score, mapped_name, cls_id))
        return pred
