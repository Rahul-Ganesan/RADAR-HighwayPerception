import cv2
import numpy as np


def render_overlay(frame, result):
    out = frame.copy()
    for tr in result.tracks:
        x1, y1, x2, y2 = map(int, tr.xyxy)
        cv2.rectangle(out, (x1, y1), (x2, y2), (40, 220, 80), 2)
        cv2.putText(out, f"{tr.class_name} #{tr.track_id} {tr.score:.2f}", (x1, max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 80), 1, cv2.LINE_AA)
    if result.lane is not None and result.lane.mask is not None:
        lane_color = np.zeros_like(out)
        lane_color[:, :, 1] = result.lane.mask.astype(np.uint8)
        out = cv2.addWeighted(out, 1.0, lane_color, 0.35, 0.0)
    cv2.putText(out, f"condition: {result.condition}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_video_writer(path: str, width: int, height: int, fps: float = 30.0):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))
