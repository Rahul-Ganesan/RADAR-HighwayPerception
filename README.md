# HighwayPerception All-Weather

All-weather highway perception pipeline for object detection, tracking, lane estimation, and condition-aware evaluation.

## Overview

This project provides:
- YOLOv8-based object detection.
- ByteTrack-style multi-object tracking adapter.
- Lane estimation adapter (model-backed when weights are available, heuristic fallback otherwise).
- Condition-sliced evaluation on BDD100K metadata (`clear`, `rain`, `fog`, `night`, etc.).

Primary use cases:
- Run end-to-end inference on a highway video.
- Evaluate detection performance by weather/time-of-day condition.

## Tech Stack

- Python `>=3.10`
- Ultralytics YOLO (`ultralytics`)
- OpenCV (`opencv-python`)
- NumPy, PyYAML, tqdm
- Optional metrics/dev extras: `motmetrics`, `scikit-learn`, `pytest`, `ruff`

## Repository Structure

```text
HighwayPerception_AllWeather/
  configs/
    models.yaml
    thresholds.yaml
  scripts/
    run_demo.py
    eval_conditions.py
  src/highway_perception/
    detector/
    tracker/
    lane/
    quality/
    metrics/
    data/
  outputs/
    demo/
    eval/
```

## Setup

### 1) Create and activate environment

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install project dependencies

From the project root:

```bash
pip install -e .[metrics,dev]
```

## Configuration

Update runtime settings in:
- `configs/models.yaml` for model backend, model path, classes, and optional lane weights.
- `configs/thresholds.yaml` for detection/lane thresholds per condition.

Default detection model is `yolov8m.pt` (with `yolov8s.pt` fallback configured).

## Data Requirements

### Demo inference

- Any readable input video file (e.g., `.mp4`).

### Condition evaluation (BDD100K)

`scripts/eval_conditions.py` expects a BDD root folder with image and metadata labels, for example:

```text
<bdd_root>/
  images/100k/val/*.jpg
  labels/bdd100k_labels_images_val.json
```

The script also checks several fallback image layouts under `<bdd_root>/images/...`.

## How To Execute

Run these commands from the repository root.

### 1) Run demo pipeline on video

```bash
python scripts/run_demo.py \
  --video <path_to_input_video.mp4> \
  --output outputs/demo/demo.mp4
```

Optional config overrides:
- `--models-cfg configs/models.yaml`
- `--thresholds-cfg configs/thresholds.yaml`

### 2) Run lightweight condition-sliced evaluation

```bash
python scripts/eval_conditions.py \
  --bdd-root <path_to_bdd100k_root> \
  --split val \
  --mode lightweight \
  --output outputs/eval
```

### 3) Run benchmark evaluation (`mAP@50`, `mAP@50:95`)

```bash
python scripts/eval_conditions.py \
  --bdd-root <path_to_bdd100k_root> \
  --split val \
  --mode benchmark \
  --output outputs/eval
```

Useful flags:
- `--max-images 500` for a fast smoke run.
- `--det-conf 0.25` and `--det-iou 0.5` to tune detector thresholds.

## Outputs

### Demo output

- Rendered video at your `--output` path (example: `outputs/demo/demo.mp4`).

### Evaluation output

- `outputs/eval/condition_metrics.json`

Payload includes:
- `run_metadata` (split, model path, thresholds, missing/unreadable images, sample counts)
- `overall` detection/tracking metrics
- `per_condition` detection metrics
- `slice_sizes` and `clean_vs_adverse_sizes`

## Metric Notes

- `lightweight` mode: IoU@0.5 precision/recall/F1 style reporting.
- `benchmark` mode: COCO-style 101-point AP with `mAP@50` and `mAP@50:95`.
- Tracking metrics are currently proxy/unavailable for BDD image-label evaluation due to missing track-level GT in this flow.

## Troubleshooting

- If YOLO weights are not found, verify internet/model download availability or set an explicit model path in `configs/models.yaml`.
- If many images are skipped, validate your BDD directory layout and `--split`.
- If CUDA is unavailable, set a CPU-compatible device/config and rerun.
