import json
from pathlib import Path


def load_bdd_metadata_samples(bdd_root: str, split: str = "val"):
    path = Path(bdd_root) / "labels" / f"bdd100k_labels_images_{split}.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def extract_condition_fields(sample):
    attrs = sample.get("attributes", {})
    return {
        "weather": str(attrs.get("weather", "unknown")).lower(),
        "timeofday": str(attrs.get("timeofday", "unknown")).lower(),
        "scene": str(attrs.get("scene", "unknown")).lower(),
    }


def iterate_detection_targets(samples):
    for sample in samples:
        for lb in sample.get("labels", []) or []:
            if "box2d" not in lb:
                continue
            yield {
                "image_name": sample.get("name"),
                "category": lb.get("category"),
                "box2d": lb.get("box2d"),
                "attributes": extract_condition_fields(sample),
            }
