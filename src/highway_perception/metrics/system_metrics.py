import numpy as np


def summarize_latency(latencies_ms, elapsed_s, frames):
    arr = np.array(latencies_ms, dtype=float) if latencies_ms else np.array([0.0], dtype=float)
    fps = 0.0 if elapsed_s <= 0 else frames / elapsed_s
    return {
        "fps": float(fps),
        "latency_p50_ms": float(np.percentile(arr, 50)),
        "latency_p95_ms": float(np.percentile(arr, 95)),
    }
