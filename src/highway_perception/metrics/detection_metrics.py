from collections import defaultdict

TARGET_CLASSES = ["car", "truck", "bus", "motor", "person", "rider"]


def _to_xyxy(box):
    if not box:
        return None
    if isinstance(box, (list, tuple)) and len(box) == 4:
        x1, y1, x2, y2 = [float(v) for v in box]
        return (x1, y1, x2, y2)
    if isinstance(box, dict):
        keys = ("x1", "y1", "x2", "y2")
        if all(k in box for k in keys):
            return tuple(float(box[k]) for k in keys)
    return None


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return 0.0 if union <= 0.0 else inter / union


def _build_gt_index(targets):
    gt = defaultdict(list)
    for t in targets:
        c = t.get("category")
        if c not in TARGET_CLASSES:
            continue
        box = _to_xyxy(t.get("box2d") or t.get("xyxy"))
        if box is None:
            continue
        gt[(str(t.get("image_name", "")), c)].append(box)
    return gt


def _build_pred_index(predictions):
    pred = defaultdict(list)
    for p in predictions:
        c = p.get("category")
        if c not in TARGET_CLASSES:
            continue
        box = _to_xyxy(p.get("box2d") or p.get("xyxy"))
        if box is None:
            continue
        score = float(p.get("score", 0.0))
        pred[(str(p.get("image_name", "")), c)].append({"box": box, "score": score})
    return pred


def _match_counts(predictions, targets, iou_thr=0.5):
    gt = _build_gt_index(targets)
    pred = _build_pred_index(predictions)
    counts = {c: {"tp": 0, "fp": 0, "fn": 0} for c in TARGET_CLASSES}
    for key in set(gt.keys()) | set(pred.keys()):
        image_name, cls_name = key
        gt_boxes = gt.get((image_name, cls_name), [])
        pred_boxes = sorted(pred.get((image_name, cls_name), []), key=lambda x: x["score"], reverse=True)
        used = [False] * len(gt_boxes)
        for p in pred_boxes:
            best_idx = -1
            best_iou = 0.0
            for i, g in enumerate(gt_boxes):
                if used[i]:
                    continue
                iou = _iou(p["box"], g)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0 and best_iou >= iou_thr:
                used[best_idx] = True
                counts[cls_name]["tp"] += 1
            else:
                counts[cls_name]["fp"] += 1
        counts[cls_name]["fn"] += sum(0 if u else 1 for u in used)
    return counts


def _safe_div(num, den):
    return 0.0 if den <= 0 else float(num) / float(den)


def _prf(tp, fp, fn):
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    return precision, recall, f1


def _coco_ap_for_class(predictions, targets, cls_name, iou_thr):
    gt_by_img = defaultdict(list)
    for t in targets:
        if t.get("category") != cls_name:
            continue
        box = _to_xyxy(t.get("box2d") or t.get("xyxy"))
        if box is not None:
            gt_by_img[str(t.get("image_name", ""))].append(box)
    pred_rows = []
    for p in predictions:
        if p.get("category") != cls_name:
            continue
        box = _to_xyxy(p.get("box2d") or p.get("xyxy"))
        if box is None:
            continue
        pred_rows.append(
            {"image_name": str(p.get("image_name", "")), "box": box, "score": float(p.get("score", 0.0))}
        )
    pred_rows.sort(key=lambda x: x["score"], reverse=True)
    n_gt = sum(len(v) for v in gt_by_img.values())
    if n_gt == 0:
        return 0.0
    used = {img: [False] * len(boxes) for img, boxes in gt_by_img.items()}
    tp_flags, fp_flags = [], []
    for pred in pred_rows:
        img = pred["image_name"]
        gt_boxes = gt_by_img.get(img, [])
        matched = used.get(img, [])
        best_idx = -1
        best_iou = 0.0
        for i, g in enumerate(gt_boxes):
            if matched[i]:
                continue
            iou = _iou(pred["box"], g)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx >= 0 and best_iou >= iou_thr:
            matched[best_idx] = True
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
    if not tp_flags:
        return 0.0
    tp_cum, fp_cum = [], []
    t_sum = 0
    f_sum = 0
    for t, f in zip(tp_flags, fp_flags):
        t_sum += t
        f_sum += f
        tp_cum.append(t_sum)
        fp_cum.append(f_sum)
    recalls = [_safe_div(t, n_gt) for t in tp_cum]
    precisions = [_safe_div(t, t + f) for t, f in zip(tp_cum, fp_cum)]
    # 101-point interpolated AP (COCO-style integration approximation).
    ap = 0.0
    for r in [i / 100.0 for i in range(101)]:
        pmax = 0.0
        for rr, pp in zip(recalls, precisions):
            if rr >= r:
                pmax = max(pmax, pp)
        ap += pmax
    return ap / 101.0


def summarize_detection(predictions, targets, mode="lightweight"):
    counts = _match_counts(predictions, targets, iou_thr=0.5)
    per_class = {}
    total_tp = total_fp = total_fn = 0
    for cls_name in TARGET_CLASSES:
        tp = counts[cls_name]["tp"]
        fp = counts[cls_name]["fp"]
        fn = counts[cls_name]["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        p, r, f1 = _prf(tp, fp, fn)
        per_class[cls_name] = {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
    p_all, r_all, f1_all = _prf(total_tp, total_fp, total_fn)
    payload = {
        "mode": mode,
        "iou_match_threshold": 0.5,
        "overall": {
            "precision": p_all,
            "recall": r_all,
            "f1": f1_all,
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
        },
        "per_class": per_class,
    }
    if mode != "benchmark":
        payload["metric_definition"] = "Greedy IoU@0.5 matching with precision/recall/F1 aggregation."
        return payload
    iou_thresholds = [round(0.5 + i * 0.05, 2) for i in range(10)]
    per_class_ap50 = {}
    per_class_ap5095 = {}
    for cls_name in TARGET_CLASSES:
        ap_vals = [_coco_ap_for_class(predictions, targets, cls_name, thr) for thr in iou_thresholds]
        per_class_ap50[cls_name] = ap_vals[0]
        per_class_ap5095[cls_name] = sum(ap_vals) / len(ap_vals)
    map50 = sum(per_class_ap50.values()) / len(TARGET_CLASSES)
    map5095 = sum(per_class_ap5095.values()) / len(TARGET_CLASSES)
    payload.update(
        {
            "mAP50": map50,
            "mAP50_95": map5095,
            "per_class_ap50": per_class_ap50,
            "per_class_ap50_95": per_class_ap5095,
            "metric_definition": "101-point interpolated AP per class at IoU 0.50 and 0.50:0.95; mean across target classes.",
        }
    )
    return payload
