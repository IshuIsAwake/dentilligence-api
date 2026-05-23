"""Ported from Experimenting/classifier_experiments/common/box_ops.py.
Greedy anchor-based union clustering for overlapping detector boxes."""
from __future__ import annotations

from dataclasses import dataclass

XYXY = tuple[float, float, float, float]


@dataclass
class MergedBox:
    xyxy: XYXY
    conf: float
    n_merged: int
    member_idx: list[int]


def iou(a: XYXY, b: XYXY) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _union_xyxy(boxes):
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return (x1, y1, x2, y2)


def merge_boxes(boxes, iou_thresh: float = 0.3):
    """boxes: [(x1,y1,x2,y2,conf), ...]. Returns [MergedBox, ...]."""
    if not boxes:
        return []
    indexed = list(enumerate(boxes))
    indexed.sort(key=lambda t: t[1][4], reverse=True)
    clusters: list[dict] = []
    for orig_idx, b in indexed:
        cand: XYXY = b[:4]
        best_ci, best_iou = -1, 0.0
        for ci, cl in enumerate(clusters):
            v = iou(cand, _union_xyxy(cl["boxes"]))
            if v >= iou_thresh and v > best_iou:
                best_iou, best_ci = v, ci
        if best_ci == -1:
            clusters.append({"boxes": [cand], "confs": [b[4]], "idx": [orig_idx]})
        else:
            clusters[best_ci]["boxes"].append(cand)
            clusters[best_ci]["confs"].append(b[4])
            clusters[best_ci]["idx"].append(orig_idx)
    out = [
        MergedBox(
            xyxy=_union_xyxy(cl["boxes"]),
            conf=max(cl["confs"]),
            n_merged=len(cl["boxes"]),
            member_idx=sorted(cl["idx"]),
        )
        for cl in clusters
    ]
    out.sort(key=lambda m: m.conf, reverse=True)
    return out
