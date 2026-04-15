from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class Detection:
    xyxy: Tuple[float, float, float, float]
    score: float
    class_name: str
    class_id: int


@dataclass
class Track:
    track_id: int
    xyxy: Tuple[float, float, float, float]
    score: float
    class_name: str


@dataclass
class LaneOutput:
    mask: np.ndarray
    lane_confidence: float


@dataclass
class FrameResult:
    detections: List[Detection] = field(default_factory=list)
    tracks: List[Track] = field(default_factory=list)
    lane: Optional[LaneOutput] = None
    condition: str = "clear"
    latency_ms: Dict[str, float] = field(default_factory=dict)
