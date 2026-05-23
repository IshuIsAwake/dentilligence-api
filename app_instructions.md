# app_instructions.md — brief for the `dentiligence-api` build

This document is the implementation brief for a **new repo**
(`dentiligence-api/`) that wraps the locked OPMD detection pipeline in
a FastAPI service deployable to Google Cloud Run, plus a static
HTML/CSS/JS frontend that calls it.

The research repo this came out of is at
`/home/ishu/Projects/AI/Oral_cancer/`. Source paths in this doc refer
to that repo.

---

## 1. What you are building

A two-stage oral-lesion screening service:

```
phone photo (multipart upload)
   │
   ▼
YOLOv8n binary lesion detector     "is there a lesion, roughly where?"
   │
   ├── no detection ─────────────► {detected: false, ...}
   │
   └── lesion box(es)
           │
           ▼
   shared crop function (pad + crop)
           │
           ▼
   EfficientNet-B0 5-class disease classifier
   optional 4-view test-time augmentation
   mean softmax across views → across boxes → argmax
           │
           ▼
   {detected: true, disease, confidence, boxes, processing_ms}
```

Two deliverables:

1. **`dentiligence-api/`** — the FastAPI service + weights + Dockerfile,
   deployed to GCP Cloud Run by the client's GCP engineer.
2. **`frontend/index.html`** — a single-file static frontend (already
   designed; will be dropped into the repo by the user). It is loaded
   into a Custom HTML block on the client's WordPress site, and also
   served from disk on the developer's laptop for local testing.

---

## 2. Locked production defaults (from `Experiment_Results.md` §17)

| component | choice |
|---|---|
| detector | fold 2 of `kfold5_geom_no_color_binary` (YOLOv8n) |
| detector conf | 0.10 |
| classifier | fold 2 of `gt_pad_0.40_b0_aug` (EfficientNet-B0) |
| serve padding | 0.20 (proportional to box dims) |
| TTA | on — 4 views (identity, hflip, rot +10°, rot −10°) |
| merge boxes | off |
| aggregation | mean post-softmax across TTA views → across boxes → argmax |
| input preprocessing | raw RGB + ImageNet normalization, 224 px (B0 default). **No LAB, no CLAHE.** |

Headline measured at this config: cond_acc 0.660 ± 0.041, sys_acc 0.597
(folds 1-4, leak-free). Don't try to "improve" anything pre-shipping —
this is the validated config.

---

## 3. Weights — what to copy and where

**Detector weights — copy all four into `weights/`:**

| source path (in research repo) | destination filename |
|---|---|
| `Experimenting/results/kfold5_geom_no_color_binary/fold_0/train/weights/best.pt` | `weights/detector_fold0.pt` |
| `Experimenting/results/kfold5_geom_no_color_binary/fold_2/train/weights/best.pt` | `weights/detector_fold2.pt` |
| `Experimenting/results/kfold5_geom_no_color_binary/fold_3/train/weights/best.pt` | `weights/detector_fold3.pt` |
| `Experimenting/results/kfold5_geom_no_color_binary/fold_4/train/weights/best.pt` | `weights/detector_fold4.pt` |

Fold 1 is **not** shipped (not exposed in the dev UI). Fold 0 is
shipped only as the "conservative" option in the dev panel — it must
never be the default.

**Classifier weights — copy one:**

| source path | destination filename |
|---|---|
| `Experimenting/classifier_experiments/results/gt_pad_0.40_b0_aug/fold_2/best.pt` | `weights/classifier.pt` |

Only this one classifier ships. The train-pad selector was dropped
from the frontend, so no other classifier variants are needed. Total
weight footprint: ~6 MB × 4 + ~17 MB ≈ 41 MB, well under GitHub's
100 MB-per-file limit. Push directly to the repo; no LFS.

**Copy commands** (run from the research repo root):

```bash
SRC=/home/ishu/Projects/AI/Oral_cancer
DST=/path/to/dentiligence-api/weights
mkdir -p "$DST"
for f in 0 2 3 4; do
  cp "$SRC/Experimenting/results/kfold5_geom_no_color_binary/fold_$f/train/weights/best.pt" \
     "$DST/detector_fold$f.pt"
done
cp "$SRC/Experimenting/classifier_experiments/results/gt_pad_0.40_b0_aug/fold_2/best.pt" \
   "$DST/classifier.pt"
```

---

## 4. Reference code to port (do not invent these from scratch)

These files in the research repo are the source of truth for the
inference pipeline. Read them and port the relevant logic into the new
repo — do not redesign.

| concern | source file |
|---|---|
| shared crop function (pad + crop) | `Experimenting/classifier_experiments/common/crops.py` |
| B0 classifier model definition + weight loading | `Experimenting/classifier_experiments/common/model_b0.py` |
| full inference pipeline (detector → TTA → classifier) | `Experimenting/classifier_experiments/phase2_pipeline.py` |
| union-box merging (greedy IoU clustering @ 0.3) | `Experimenting/classifier_experiments/common/box_ops.py` |
| visualisation reference (boxes on image) | `Experimenting/inspect_pipeline.py` |

The shared crop function is **load-bearing** — it must be identical
between training and serving. See `CLAUDE.md` in the research repo
for the full invariant list.

---

## 5. Repo structure

```
dentiligence-api/
├── main.py              FastAPI app, routes, CORS, response shaping
├── inference.py         detect → crop → (merge?) → TTA classify → result
├── models.py            B0 classifier nn.Module + load helper
├── crops.py             shared pad_and_crop (ported from common/crops.py)
├── box_ops.py           union-box clustering (ported from common/box_ops.py)
├── config.py            paths, defaults, class names
├── weights/
│   ├── detector_fold0.pt
│   ├── detector_fold2.pt
│   ├── detector_fold3.pt
│   ├── detector_fold4.pt
│   └── classifier.pt
├── frontend/
│   └── index.html       static frontend (provided by user; do not modify)
├── Dockerfile
├── requirements.txt
├── .dockerignore
├── .gitignore
├── examples/
│   └── curl.sh          one-line sample call for the GCP engineer
├── tests/
│   └── test_inference.py  smoke test on a couple of bundled sample images
├── app_instructions.md  (this file)
└── README.md            deployment steps for the GCP engineer
```

Keep files small and single-purpose. No layers of abstraction beyond
what is needed for this service.

---

## 6. Disease class mapping

YOLO label `class_id` ∈ {0..4}. The detector is binary (everything
collapses to `0 = lesion`), but the classifier's softmax index follows
this exact order:

| index | disease | display name |
|---|---|---|
| 0 | Leukoplakia | Leukoplakia |
| 1 | Erythroplakia | Erythroplakia |
| 2 | OSMF | Oral Submucous Fibrosis |
| 3 | Lichen_Planus | Lichen Planus |
| 4 | NH_Ulcers | Non-healing Ulcer |

**There is no `Normal` class.** If the detector finds nothing, the
response is `{detected: false}` and the frontend renders the
"looks fine" path. Never add a 6th class.

---

## 7. API contract

### `POST /predict`

Multipart form upload + optional dev-panel overrides as form fields or
JSON.

**Request:**

| field | type | default | notes |
|---|---|---|---|
| `image` | file (multipart) | required | jpeg/png |
| `detector_fold` | int | `2` | one of {0, 2, 3, 4} |
| `detector_conf` | float | `0.10` | one of {0.01, 0.05, 0.10, 0.15, 0.25, 0.40, 0.50} — validate server-side, reject others |
| `padding` | float | `0.20` | one of {0.0, 0.2, 0.4} |
| `tta` | bool | `true` | |
| `merge_boxes` | bool | `false` | |

**Response (lesion detected):**

```json
{
  "detected": true,
  "disease": "OSMF",
  "disease_display": "Oral Submucous Fibrosis",
  "confidence": 0.78,
  "boxes": [
    {"x1": 120, "y1": 240, "x2": 380, "y2": 410, "conf": 0.42}
  ],
  "processing_ms": 850
}
```

**Response (no detection):**

```json
{
  "detected": false,
  "disease": null,
  "confidence": null,
  "boxes": [],
  "processing_ms": 420
}
```

`boxes` are in **original image pixel coordinates** (the frontend
draws them on the displayed image, possibly scaled). `confidence` is
the classifier's mean softmax for the predicted class after TTA and
box aggregation — surface this in the frontend's dev panel as
"classifier confidence". Detector box `conf` is the raw YOLO score
per box.

### `GET /health`

```json
{
  "status": "ok",
  "model_loaded": true,
  "detector_folds_loaded": [0, 2, 3, 4]
}
```

Used by Cloud Run health checks and the frontend's "API: local / prod"
indicator.

---

## 8. Inference algorithm (exact)

```
load detector_fold{N} (cached per fold in process memory; lazy-load on first request)
load classifier (cached once at startup)

predict(image, detector_fold, detector_conf, padding, tta, merge_boxes):
    boxes = detector.predict(image, conf=detector_conf)
    if not boxes:
        return {detected: false, ...}

    if merge_boxes:
        boxes = union_cluster(boxes, iou_thresh=0.3)   # box_ops.py

    per_box_softmax = []
    for box in boxes:
        crop = pad_and_crop(image, box, pad=padding)   # crops.py
        if tta:
            views = [crop, hflip(crop), rotate(crop, +10), rotate(crop, -10)]
        else:
            views = [crop]
        per_view_softmax = [softmax(classifier(preprocess(v))) for v in views]
        per_box_softmax.append(mean(per_view_softmax, axis=0))

    final = mean(per_box_softmax, axis=0)
    pred_idx = argmax(final)
    return {
        detected: true,
        disease: CLASS_NAMES[pred_idx],
        confidence: float(final[pred_idx]),
        boxes: [box.to_xyxy_pixels() for box in boxes],
        processing_ms: elapsed
    }
```

Notes:
- All averaging is **post-softmax**, not on logits.
- `preprocess` = resize to 224, ImageNet mean/std normalize, raw RGB.
- Load the four detector weights lazily on first use per fold; do not
  load all four at process startup (~24 MB extra resident for the
  three rarely-used folds × concurrent Cloud Run instances adds up).
- Run inference in `torch.inference_mode()`. CPU is fine for Cloud Run
  — GPU is not worth the cost at this latency budget.

---

## 9. CORS

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://<client-wordpress-domain>",   # placeholder — TBD
        "http://localhost",
        "http://localhost:8000",
        "http://127.0.0.1",
        "null",   # required for file:// (the developer's local HTML)
    ],
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)
```

The WordPress origin is unknown until deploy; leave a clearly-marked
placeholder in `config.py` and the README.

---

## 10. Frontend integration

The single-file HTML at `frontend/index.html` is provided by the user
— **do not modify it**. It is the design output, hand-styled. The
backend's only job is to honor the API contract above.

The frontend picks its API base URL by hostname (auto-detect):

```js
const API_URL = window.location.hostname === "localhost"
  ? "http://localhost:8080"
  : "https://<cloud-run-url>";
```

So the frontend works in two modes with no code changes:

- **Local mode** — developer runs `uvicorn main:app --port 8080` on
  laptop, opens `frontend/index.html` from disk or via a tiny static
  server. CORS allows `file://` (origin `null`) and `localhost`.
- **Prod mode** — same HTML pasted into a Custom HTML block on the
  WordPress site. Browser hostname is the WP domain, so it hits Cloud
  Run.

The dev-options panel in the frontend maps 1:1 to the `POST /predict`
form fields above. The classifier-confidence readout at the bottom of
that panel binds to `response.confidence`.

---

## 11. Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# system deps for opencv / pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
CMD exec uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1
```

`requirements.txt` (pin versions):

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
ultralytics==8.3.0
torch==2.4.1
torchvision==0.19.1
pillow==10.4.0
numpy==1.26.4
opencv-python-headless==4.10.0.84
```

`--workers 1` because the models are loaded into process memory; more
workers means more resident copies. Cloud Run scales by spawning more
instances, not more workers per instance.

---

## 12. GCP deployment (for the README)

The README in the new repo is for the **GCP engineer**, not for me.
Keep it terse:

```bash
# one-time
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID
gcloud services enable run.googleapis.com artifactregistry.googleapis.com

# build + deploy
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/dentiligence-api
gcloud run deploy dentiligence-api \
  --image gcr.io/YOUR_PROJECT_ID/dentiligence-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 60 \
  --max-instances 5
```

`--memory 4Gi` because B0 + YOLOv8n + Torch resident is ~1.5-2 GB and
we want headroom for concurrent requests. `--cpu 2` because inference
is CPU-bound and a single request benefits from 2 cores on Torch
ops. `--max-instances 5` is a cost cap; raise if traffic justifies.

---

## 13. What NOT to do

Inherited from `CLAUDE.md` in the research repo. These are binding.

- **No `Normal` class in the classifier.** Healthy = detector silent.
- **No LAB / CLAHE preprocessing.** Raw RGB + ImageNet normalize only.
- **Do not modify the shared crop function** without re-validating
  against the §17 headline.
- **Do not ship detector fold 1** (not exposed; not measured at the
  serving threshold the same way).
- **Do not change the TTA recipe** (identity + hflip + rot ±10°). It
  was chosen to stay inside the rotation range the classifier saw at
  training. Adding vflip changes oral anatomy semantics.
- **Do not ensemble the 4 detector folds.** Each test image was in
  4/5 folds' training sets — no honest eval exists for an ensemble.
  Folds are user-selectable, not co-applied.
- **Do not invent new endpoints / fields** beyond §7. The frontend is
  already designed against this contract.

---

## 14. Done = ?

- `dentiligence-api/` builds locally: `docker build -t opmd .`
  then `docker run -p 8080:8080 opmd` returns `/health` ok.
- `POST /predict` with a sample image from
  `Experimenting/internet_images/` in the research repo returns a
  sensible JSON response in <2 s on the developer's laptop CPU.
- `frontend/index.html` opened from disk (file://) successfully calls
  the local backend and renders the result card.
- Smoke test in `tests/test_inference.py` passes on at least 2 known
  positives and 1 known negative.
- README contains the gcloud command block above with `YOUR_PROJECT_ID`
  placeholders the engineer can fill in.

No deploy from this repo. The engineer deploys.
