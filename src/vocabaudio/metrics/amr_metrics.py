from __future__ import annotations

from collections import defaultdict

from vocabaudio.utils.windows import normalize_windows, temporal_iou


def _ranked_matches(
    gt_windows: list[list[float]],
    pred_windows: list[list[float]],
    threshold: float,
) -> list[int]:
    gt_used = [False] * len(gt_windows)
    matches = []
    pred_sorted = sorted(pred_windows, key=lambda x: float(x[2]) if len(x) > 2 else 1.0, reverse=True)
    for pred in pred_sorted:
        best_idx = -1
        best_iou = 0.0
        for idx, gt in enumerate(gt_windows):
            if gt_used[idx]:
                continue
            iou = temporal_iou(gt, pred)
            if iou >= threshold and iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_idx >= 0:
            gt_used[best_idx] = True
            matches.append(1)
        else:
            matches.append(0)
    return matches


def _average_precision(matches: list[int], num_gt: int) -> float:
    if num_gt <= 0 or not matches:
        return 0.0
    tp = 0
    precisions = []
    for rank, matched in enumerate(matches, start=1):
        if matched:
            tp += 1
            precisions.append(tp / rank)
    if not precisions:
        return 0.0
    return sum(precisions) / num_gt


def evaluate_predictions(rows: list[dict], predictions: list[dict]) -> dict:
    pred_by_qid = {row["qid"]: row for row in predictions}
    thresholds = [0.5, 0.7, 0.75]
    r1_scores = defaultdict(list)
    ap_scores = defaultdict(list)

    for row in rows:
        gt_windows = normalize_windows(row["relevant_windows"], float(row["duration"]))
        pred = pred_by_qid.get(row["qid"], {"pred_relevant_windows": []})
        pred_windows = pred.get("pred_relevant_windows", [])
        pred_windows = [list(window) for window in pred_windows]
        if not pred_windows:
            pred_windows = []

        for thr in thresholds:
            matches = _ranked_matches(gt_windows, pred_windows, thr)
            r1_scores[thr].append(float(matches[0]) if matches else 0.0)
            ap_scores[thr].append(_average_precision(matches, len(gt_windows)))

    metrics = {
        "R1@0.5": round(100.0 * sum(r1_scores[0.5]) / max(len(r1_scores[0.5]), 1), 2),
        "R1@0.7": round(100.0 * sum(r1_scores[0.7]) / max(len(r1_scores[0.7]), 1), 2),
        "mAP@0.5": round(100.0 * sum(ap_scores[0.5]) / max(len(ap_scores[0.5]), 1), 2),
        "mAP@0.75": round(100.0 * sum(ap_scores[0.75]) / max(len(ap_scores[0.75]), 1), 2),
    }
    avg_ap = (metrics["mAP@0.5"] + metrics["mAP@0.75"]) / 2.0
    metrics["mAP@avg"] = round(avg_ap, 2)
    metrics["brief"] = (
        f"R1@0.5={metrics['R1@0.5']:.2f}, "
        f"R1@0.7={metrics['R1@0.7']:.2f}, "
        f"mAP@avg={metrics['mAP@avg']:.2f}"
    )
    return metrics

