import argparse
import json
from pathlib import Path
import cv2
import yaml
from tqdm import tqdm

from highway_perception.data.bdd_loader import iterate_detection_targets, load_bdd_metadata_samples
from highway_perception.data.splits import build_condition_slices, merge_clean_vs_adverse
from highway_perception.metrics.detection_metrics import summarize_detection
from highway_perception.metrics.tracking_metrics import summarize_tracking
from highway_perception.detector.yolo_ultralytics import YOLOUltralyticsDetector


def resolve_bdd_image_path(bdd_root, split, image_name):
    candidates = [
        Path(bdd_root) / "images" / "100k" / split / image_name,
        Path(bdd_root) / "images" / split / image_name,
        Path(bdd_root) / "images" / image_name,
        Path(bdd_root) / image_name,
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _build_condition_records(samples):
    records = []
    for s in samples:
        attrs = s.get("attributes", {})
        weather = str(attrs.get("weather", "unknown")).lower()
        tod = str(attrs.get("timeofday", "unknown")).lower()
        condition = "night" if tod == "night" else weather
        records.append({"image_name": s.get("name"), "condition": condition, "attributes": attrs})
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bdd-root", required=True)
    p.add_argument("--acdc-root", default="")
    p.add_argument("--split", default="val")
    p.add_argument("--models-cfg", default="configs/models.yaml")
    p.add_argument("--mode", choices=["lightweight", "benchmark"], default="lightweight")
    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--det-conf", type=float, default=0.25)
    p.add_argument("--det-iou", type=float, default=0.5)
    p.add_argument("--output", default="outputs/eval")
    args = p.parse_args()

    with open(args.models_cfg, "r", encoding="utf-8") as f:
        models_cfg = yaml.safe_load(f)
    det_cfg = models_cfg.get("detection", {})
    detector = YOLOUltralyticsDetector(
        model_path=det_cfg.get("model", "yolov8m.pt"),
        class_names=det_cfg.get("classes", []),
        conf=args.det_conf,
        iou=args.det_iou,
    )

    samples = load_bdd_metadata_samples(args.bdd_root, split=args.split)
    if args.max_images > 0:
        samples = samples[: args.max_images]
    targets = list(iterate_detection_targets(samples))

    predictions = []
    missing_images = 0
    unreadable_images = 0
    for sample in tqdm(samples, desc="Running detector inference", unit="img"):
        image_name = sample.get("name")
        img_path = resolve_bdd_image_path(args.bdd_root, args.split, image_name)
        if img_path is None:
            missing_images += 1
            continue
        frame = cv2.imread(str(img_path))
        if frame is None:
            unreadable_images += 1
            continue
        for det in detector.predict(frame):
            predictions.append(
                {
                    "image_name": image_name,
                    "category": det.class_name,
                    "xyxy": list(det.xyxy),
                    "score": float(det.score),
                }
            )

    det_metrics = summarize_detection(predictions, targets, mode=args.mode)
    trk_metrics = summarize_tracking(track_records=[], has_tracking_gt=False)
    records = _build_condition_records(samples)
    if args.acdc_root:
        # Optional extension: ACDC records can be mixed in for condition distribution reporting.
        pass
    slices = build_condition_slices(records)
    grouped = merge_clean_vs_adverse(slices)

    per_condition_detection = {}
    for cond, recs in slices.items():
        image_names = {r.get("image_name") for r in recs if r.get("image_name")}
        cond_targets = [t for t in targets if t.get("image_name") in image_names]
        cond_predictions = [p for p in predictions if p.get("image_name") in image_names]
        per_condition_detection[cond] = summarize_detection(cond_predictions, cond_targets, mode=args.mode)

    payload = {
        "run_metadata": {
            "split": args.split,
            "mode": args.mode,
            "models_cfg": args.models_cfg,
            "model_path": det_cfg.get("model", "yolov8m.pt"),
            "num_samples": len(samples),
            "num_targets": len(targets),
            "num_predictions": len(predictions),
            "missing_images": missing_images,
            "unreadable_images": unreadable_images,
            "det_conf": args.det_conf,
            "det_iou": args.det_iou,
        },
        "overall": {"detection": det_metrics, "tracking": trk_metrics},
        "per_condition": {"detection": per_condition_detection},
        "tracking": trk_metrics,
        "slice_sizes": {k: len(v) for k, v in slices.items()},
        "clean_vs_adverse_sizes": {k: len(v) for k, v in grouped.items()},
    }
    out = Path(args.output) / "condition_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
