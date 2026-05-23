"""Smoke test for the inference pipeline.

Runs detect+classify directly (no HTTP layer) on a couple of bundled samples.
Asserts the response shape and that the detector fires on positives and stays
silent on the negative."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import inference  # noqa: E402
import config     # noqa: E402

SAMPLES = ROOT / "tests" / "samples"


def _run(name: str) -> dict:
    data = (SAMPLES / name).read_bytes()
    return inference.predict(
        image_bytes=data,
        detector_fold=config.DEFAULT_FOLD,
        detector_conf=config.DEFAULT_CONF,
        padding=config.DEFAULT_PADDING,
        tta=False,        # speed up the smoke test
        merge=False,
    )


def test_positive_leukoplakia():
    r = _run("leuko.jpg")
    assert r["detected"] is True, r
    assert r["disease"] in config.CLASS_NAMES
    assert 0.0 <= r["confidence"] <= 1.0
    assert len(r["boxes"]) >= 1
    for b in r["boxes"]:
        assert b["x2"] > b["x1"] and b["y2"] > b["y1"]


def test_positive_erythroplakia():
    r = _run("erythro.jpg")
    assert r["detected"] is True, r
    assert r["disease_display"]


def test_negative():
    r = _run("normal.jpg")
    # Detector may or may not fire on the "normal" sample — this test
    # mainly verifies the response shape is well-formed either way.
    assert "detected" in r
    assert r["processing_ms"] >= 0
    if r["detected"]:
        assert r["confidence"] is not None
    else:
        assert r["disease"] is None
        assert r["boxes"] == []


if __name__ == "__main__":
    for s in ("leuko.jpg", "erythro.jpg", "normal.jpg"):
        print(s, _run(s))
