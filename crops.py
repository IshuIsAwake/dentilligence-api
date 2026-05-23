"""Ported from Experimenting/classifier_experiments/common/crops.py.
The pad-and-crop geometry must stay byte-equivalent to training."""
from __future__ import annotations

import numpy as np


def pad_and_crop(bgr: np.ndarray, xyxy, pad_frac: float):
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = xyxy
    w = x2 - x1
    h = y2 - y1
    px = pad_frac * w
    py = pad_frac * h
    nx1 = int(max(0, round(x1 - px)))
    ny1 = int(max(0, round(y1 - py)))
    nx2 = int(min(W, round(x2 + px)))
    ny2 = int(min(H, round(y2 + py)))
    if nx2 <= nx1 or ny2 <= ny1:
        return None
    return bgr[ny1:ny2, nx1:nx2].copy()
