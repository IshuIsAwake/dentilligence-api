"""FastAPI app exposing the OPMD detection pipeline.

POST /predict — multipart image + dev-panel overrides → JSON result.
GET  /health  — liveness for Cloud Run + frontend API indicator.
"""
from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import config
import inference

app = FastAPI(title="dentiligence-api", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    inference.warm_classifier()


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": True,
        "detector_folds_loaded": inference.loaded_folds(),
    }


@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    detector_fold: int = Form(config.DEFAULT_FOLD),
    detector_conf: float = Form(config.DEFAULT_CONF),
    padding: float = Form(config.DEFAULT_PADDING),
    tta: bool = Form(config.DEFAULT_TTA),
    merge_boxes: bool = Form(config.DEFAULT_MERGE),
):
    if detector_fold not in config.ALLOWED_FOLDS:
        raise HTTPException(400, f"detector_fold must be one of {sorted(config.ALLOWED_FOLDS)}")
    if not any(abs(detector_conf - v) < 1e-9 for v in config.ALLOWED_CONFS):
        raise HTTPException(400, f"detector_conf must be one of {sorted(config.ALLOWED_CONFS)}")
    if not any(abs(padding - v) < 1e-9 for v in config.ALLOWED_PADDINGS):
        raise HTTPException(400, f"padding must be one of {sorted(config.ALLOWED_PADDINGS)}")

    data = await image.read()
    if not data:
        raise HTTPException(400, "empty image upload")

    try:
        result = inference.predict(
            image_bytes=data,
            detector_fold=detector_fold,
            detector_conf=detector_conf,
            padding=padding,
            tta=tta,
            merge=merge_boxes,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return JSONResponse(result)
