from typing import Dict, List, Tuple
from highway_perception.types import Detection, Track


class ByteTrackAdapter:
    def __init__(self, match_thresh: float = 0.3):
        self.match_thresh = match_thresh
        self.next_id = 1
        self._tracks: Dict[int, Tuple[float, float, float, float]] = {}

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return 0.0 if union <= 0 else inter / union

    def update(self, detections: List[Detection]) -> List[Track]:
        assigned_tracks = set()
        out: List[Track] = []
        for det in detections:
            best_id, best_iou = None, 0.0
            for tid, tbox in self._tracks.items():
                if tid in assigned_tracks:
                    continue
                iou = self._iou(det.xyxy, tbox)
                if iou > best_iou:
                    best_iou, best_id = iou, tid
            if best_id is None or best_iou < self.match_thresh:
                best_id = self.next_id
                self.next_id += 1
            self._tracks[best_id] = det.xyxy
            assigned_tracks.add(best_id)
            out.append(Track(best_id, det.xyxy, det.score, det.class_name))
        self._tracks = {tid: self._tracks[tid] for tid in assigned_tracks}
        return out
