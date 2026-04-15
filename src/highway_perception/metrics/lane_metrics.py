import numpy as np


def lane_f1_iou(pred_mask, gt_mask):
    pred = pred_mask > 0
    gt = gt_mask > 0
    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    return {"F1": f1, "IoU": iou, "FP": fp, "FN": fn, "Accuracy": (tp / (gt.sum() + 1e-6))}
