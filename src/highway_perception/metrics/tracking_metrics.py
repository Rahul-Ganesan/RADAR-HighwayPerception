def summarize_tracking(track_records=None, has_tracking_gt=False):
    if has_tracking_gt:
        return {
            "status": "not_implemented",
            "metric_definition": "Ground-truth tracking evaluation path is not implemented in this MVP.",
        }
    count = len(track_records or [])
    return {
        "status": "proxy_not_available",
        "records_seen": count,
        "metric_definition": "BDD100K image-label split has no track GT in this pipeline. MOT metrics (HOTA/MOTA/IDF1) are not reported.",
    }
