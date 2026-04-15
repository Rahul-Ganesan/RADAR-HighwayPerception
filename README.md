# HighwayPerception All-Weather MVP

MVP perception stack for highway scenes under clean and adverse conditions.

## Components

- Detection: Ultralytics YOLOv8 (`yolov8m.pt` default, `yolov8s.pt` fallback)
- Tracking: ByteTrack-style tracker adapter (with fallback matching)
- Lane estimation: LaneAF adapter (model-backed when available, heuristic fallback)
- Condition-aware analysis: BDD100K metadata and ACDC condition slices

## Quick Start

1. Install:

```bash
pip install -e .[metrics,dev]
```

2. Configure model/data paths in `configs/models.yaml`.

3. Run demo video generation:

```bash
python scripts/run_demo.py --video <input.mp4> --output outputs/demo/demo.mp4
```

4. Run condition-sliced evaluation (lightweight real-inference mode):

```bash
python scripts/eval_conditions.py --bdd-root <path_to_bdd100k> --split val --mode lightweight --output outputs/eval
```

5. Run benchmark-grade detection metrics (`mAP@50`, `mAP@50:95`) on the same split:

```bash
python scripts/eval_conditions.py --bdd-root <path_to_bdd100k> --split val --mode benchmark --output outputs/eval
```

## Evaluation Notes

- `lightweight` mode reports IoU@0.5 precision/recall/F1 (overall + per class + per weather condition).
- `benchmark` mode adds COCO-style 101-point interpolated AP with `mAP@50` and `mAP@50:95`.
- Output file: `outputs/eval/condition_metrics.json` with:
  - `run_metadata` (model path, split size, thresholds, read failures),
  - `overall` metrics,
  - `per_condition` detection metrics,
  - normalized `slice_sizes` and `clean_vs_adverse_sizes`.
- Tracking metrics are intentionally marked as unavailable/proxy for BDD image labels, because this split does not include track ground truth in this pipeline.
