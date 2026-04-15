from pathlib import Path

ACDC_CONDITIONS = ("fog", "night", "rain", "snow")


def discover_acdc_images(acdc_root: str):
    root = Path(acdc_root)
    records = []
    for condition in ACDC_CONDITIONS:
        cond_dir = root / condition
        if not cond_dir.exists():
            continue
        for img in cond_dir.rglob("*.png"):
            records.append({"path": str(img), "condition": condition})
    return records
