"""Paths, defaults, class names. Single source of truth for the service config."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"

DETECTOR_WEIGHTS = {
    0: WEIGHTS_DIR / "detector_fold0.pt",
    2: WEIGHTS_DIR / "detector_fold2.pt",
    3: WEIGHTS_DIR / "detector_fold3.pt",
    4: WEIGHTS_DIR / "detector_fold4.pt",
}
CLASSIFIER_WEIGHTS = WEIGHTS_DIR / "classifier.pt"

ALLOWED_FOLDS = {0, 2, 3, 4}
ALLOWED_CONFS = {0.01, 0.05, 0.10, 0.15, 0.25, 0.40, 0.50}
ALLOWED_PADDINGS = {0.0, 0.2, 0.4}

DEFAULT_FOLD = 2
DEFAULT_CONF = 0.10
DEFAULT_PADDING = 0.20
DEFAULT_TTA = True
DEFAULT_MERGE = False

INPUT_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.225, 0.224)

# Disease class mapping — softmax index → (api_id, display_name).
# Matches Experimenting/classifier_experiments/common/settings.ORIG_ID_TO_CLASS.
CLASS_NAMES = ["Leukoplakia", "Erythroplakia", "OSMF", "Lichen_Planus", "NH_Ulcers"]
DISPLAY_NAMES = {
    "Leukoplakia": "Leukoplakia",
    "Erythroplakia": "Erythroplakia",
    "OSMF": "Oral Submucous Fibrosis",
    "Lichen_Planus": "Lichen Planus",
    "NH_Ulcers": "Non-healing Ulcer",
}

# TODO: replace with the client's actual WordPress origin before deploy.
WORDPRESS_ORIGIN = "https://example.com"
CORS_ORIGINS = [
    WORDPRESS_ORIGIN,
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8080",
    "null",  # file:// origin
]
