from collections import defaultdict


def normalize_condition_key(raw_key):
    key = str(raw_key or "unknown").strip().lower()
    aliases = {
        "sunny": "clear",
        "daytime": "day",
        "partly cloudy": "overcast",
        "cloudy": "overcast",
        "rainy": "rain",
        "snowy": "snow",
        "foggy": "fog",
    }
    return aliases.get(key, key)


def build_condition_slices(records):
    slices = defaultdict(list)
    for rec in records:
        condition = rec.get("condition")
        if not condition:
            weather = rec.get("attributes", {}).get("weather", "unknown")
            tod = rec.get("attributes", {}).get("timeofday", "unknown")
            condition = "night" if tod == "night" else weather
        slices[normalize_condition_key(condition)].append(rec)
    return dict(slices)


def merge_clean_vs_adverse(slices):
    clean_keys = {"clear", "day", "overcast"}
    adverse_keys = {"rain", "fog", "snow", "night", "glare", "storm"}
    clean, adverse = [], []
    for key, vals in slices.items():
        normalized = normalize_condition_key(key)
        if normalized in clean_keys:
            clean.extend(vals)
        elif normalized in adverse_keys:
            adverse.extend(vals)
    return {"clean": clean, "adverse": adverse}
