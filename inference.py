"""Inference pipeline: detect → (merge) → crop → TTA → classify → aggregate.

Models are cached in process memory. Detector folds are lazy-loaded on first
use per fold; the classifier loads on first call. Run under torch.inference_mode.
"""
from __future__ import annotations

import threading
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

import config
from box_ops import merge_boxes
from crops import pad_and_crop
from models import load_classifier

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_classifier = None
_detectors: dict[int, Any] = {}
_lock = threading.Lock()

_preprocess = transforms.Compose([
    transforms.Resize(config.INPUT_SIZE),
    transforms.CenterCrop(config.INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
])


def device() -> torch.device:
    return _device


def get_classifier():
    global _classifier
    if _classifier is None:
        with _lock:
            if _classifier is None:
                _classifier = load_classifier(config.CLASSIFIER_WEIGHTS, _device)
    return _classifier


def get_detector(fold: int):
    if fold not in config.ALLOWED_FOLDS:
        raise ValueError(f"detector_fold {fold} not allowed")
    det = _detectors.get(fold)
    if det is None:
        with _lock:
            det = _detectors.get(fold)
            if det is None:
                from ultralytics import YOLO
                det = YOLO(str(config.DETECTOR_WEIGHTS[fold]))
                _detectors[fold] = det
    return det


def loaded_folds() -> list[int]:
    return sorted(_detectors.keys())


def warm_classifier() -> bool:
    get_classifier()
    return True


def _tta_views(pil: Image.Image) -> list[Image.Image]:
    return [pil, TF.hflip(pil), TF.rotate(pil, 10), TF.rotate(pil, -10)]


def _classify_crops(crops_bgr: list[np.ndarray], use_tta: bool) -> np.ndarray:
    model = get_classifier()
    per_crop = []
    with torch.inference_mode():
        for crop in crops_bgr:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            views = _tta_views(pil) if use_tta else [pil]
            x = torch.stack([_preprocess(v) for v in views]).to(_device)
            probs = F.softmax(model(x), dim=1).mean(dim=0).cpu().numpy()
            per_crop.append(probs)
    return np.mean(per_crop, axis=0)


def _detect(detector, bgr: np.ndarray, conf: float):
    res = detector.predict(source=bgr, conf=conf, verbose=False)
    out = []
    for r in res:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            out.append((float(x1), float(y1), float(x2), float(y2), float(c)))
    return out


def predict(image_bytes: bytes, detector_fold: int, detector_conf: float,
            padding: float, tta: bool, merge: bool) -> dict:
    t0 = time.perf_counter()
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image")

    detector = get_detector(detector_fold)
    raw_boxes = _detect(detector, bgr, detector_conf)

    if not raw_boxes:
        return {
            "detected": False,
            "disease": None,
            "disease_display": None,
            "confidence": None,
            "boxes": [],
            "processing_ms": int((time.perf_counter() - t0) * 1000),
        }

    if merge:
        merged = merge_boxes(raw_boxes, iou_thresh=0.3)
        boxes_for_crop = [(m.xyxy[0], m.xyxy[1], m.xyxy[2], m.xyxy[3], m.conf)
                          for m in merged]
    else:
        boxes_for_crop = raw_boxes

    crops, kept_boxes = [], []
    for b in boxes_for_crop:
        c = pad_and_crop(bgr, b[:4], padding)
        if c is not None:
            crops.append(c)
            kept_boxes.append(b)

    if not crops:
        return {
            "detected": False,
            "disease": None,
            "disease_display": None,
            "confidence": None,
            "boxes": [],
            "processing_ms": int((time.perf_counter() - t0) * 1000),
        }

    mean_probs = _classify_crops(crops, use_tta=tta)
    pred_idx = int(np.argmax(mean_probs))
    disease = config.CLASS_NAMES[pred_idx]

    return {
        "detected": True,
        "disease": disease,
        "disease_display": config.DISPLAY_NAMES[disease],
        "confidence": float(mean_probs[pred_idx]),
        "boxes": [
            {"x1": float(b[0]), "y1": float(b[1]),
             "x2": float(b[2]), "y2": float(b[3]), "conf": float(b[4])}
            for b in kept_boxes
        ],
        "processing_ms": int((time.perf_counter() - t0) * 1000),
    }
