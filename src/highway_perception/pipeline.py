import time
from pathlib import Path
import cv2
import yaml
from highway_perception.detector.yolo_ultralytics import YOLOUltralyticsDetector
from highway_perception.lane.laneaf_adapter import LaneAFAdapter
from highway_perception.quality.condition_classifier import ConditionClassifier
from highway_perception.tracker.bytetrack_adapter import ByteTrackAdapter
from highway_perception.types import FrameResult
from highway_perception.viz.overlay import make_video_writer, render_overlay


class HighwayPerceptionPipeline:
    def __init__(self, models_cfg, thresh_cfg):
        det_cfg = models_cfg["detection"]
        self.detector = YOLOUltralyticsDetector(det_cfg["model"], det_cfg["classes"],
                                                conf=thresh_cfg["default"]["det_conf"],
                                                iou=thresh_cfg["default"]["nms_iou"])
        self.tracker = ByteTrackAdapter(match_thresh=models_cfg["tracking"].get("match_thresh", 0.3))
        self.lane = LaneAFAdapter(models_cfg["lane"].get("weights", ""))
        self.quality = ConditionClassifier(models_cfg.get("quality", {}).get("labels"))
        self.thresh_cfg = thresh_cfg

    @classmethod
    def from_files(cls, models_path, thresh_path):
        with open(models_path, "r", encoding="utf-8") as f:
            m = yaml.safe_load(f)
        with open(thresh_path, "r", encoding="utf-8") as f:
            t = yaml.safe_load(f)
        return cls(m, t)

    def process_frame(self, frame):
        result = FrameResult()
        condition = self.quality.top_label(frame)
        result.condition = condition
        thresholds = self.thresh_cfg.get("by_condition", {}).get(condition, self.thresh_cfg["default"])
        self.detector.conf = thresholds["det_conf"]
        self.detector.iou = thresholds["nms_iou"]
        t0 = time.perf_counter()
        result.detections = self.detector.predict(frame)
        result.latency_ms["detection"] = (time.perf_counter() - t0) * 1000.0
        result.tracks = self.tracker.update(result.detections)
        result.lane = self.lane.predict(frame, conf_thresh=thresholds["lane_conf"])
        return result

    def run_video(self, input_video, output_video):
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open input video: {input_video}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        Path(output_video).parent.mkdir(parents=True, exist_ok=True)
        writer = make_video_writer(output_video, width, height, fps)
        n_frames = 0
        t_start = time.perf_counter()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            n_frames += 1
            writer.write(render_overlay(frame, self.process_frame(frame)))
        cap.release()
        writer.release()
        elapsed = max(1e-6, time.perf_counter() - t_start)
        return {"frames": n_frames, "elapsed_s": elapsed, "fps": n_frames / elapsed}
