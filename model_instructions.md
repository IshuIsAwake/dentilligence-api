# Model integration instructions

How to wire the shipped ONNX weights into any backend. This document
describes **what the models expect and what they return** — exact tensor
shapes, dtypes, value ranges, and the steps the surrounding code is
responsible for. It does not prescribe an implementation; you are free
to use whichever ONNX runtime (onnxruntime-node, onnxruntime-web,
onnxruntime, ORT-Java, ORT-Go, Triton, …) and image library suits your
stack.

For background on what these models do and how accurate they are, read
[MODEL_SUMMARY.md](MODEL_SUMMARY.md). For the operational behaviour the
service is expected to expose, read [app_instructions.md](app_instructions.md).
For a reference implementation of the same pipeline in Python, read
[inference.py](inference.py) (informative, not required).

---

## 1. Files you have

In [weights/](weights/):

| file | role | size |
|---|---|---|
| `detector_fold2.onnx` | YOLOv8n binary lesion detector (production default) | ~12 MB |
| `detector_fold0.onnx` | YOLOv8n, fold 0 ("Conservative") | ~12 MB |
| `detector_fold3.onnx` | YOLOv8n, fold 3 ("Alternative") | ~12 MB |
| `detector_fold4.onnx` | YOLOv8n, fold 4 ("Aggressive") | ~12 MB |
| `classifier.onnx`     | EfficientNet-B0 5-class disease classifier | ~17 MB |

The `.pt` files are PyTorch state-dicts of the same weights, shipped
only for teams already on a PyTorch stack. **Prefer the ONNX files** —
they are framework-agnostic and self-contained.

The `classifier.onnx.data` file is a legacy external-weights sidecar
from an earlier export and is no longer referenced by `classifier.onnx`
(weights are embedded). You can ignore or delete it.

---

## 2. The pipeline you must build

For every uploaded image, in order:

```
image bytes (JPEG/PNG)
  │
  ▼
[A] decode + letterbox to 640×640 RGB float32 ─► run detector ONNX
  │
  ▼
[B] decode detector output: 1×5×8400 → list of (x1,y1,x2,y2,conf) boxes
  │
  ▼
[C] filter by confidence threshold, run NMS
  │
  ▼
   no boxes ─► return {detected:false}
  │
  ▼
[D] map boxes back to original image coordinates (undo letterbox)
  │
  ▼
[E] (optional) merge overlapping boxes
  │
  ▼
[F] for each box: pad by P, crop the original image
  │
  ▼
[G] for each crop: resize→center-crop→normalize to 224×224, optionally
    generate TTA views, run classifier ONNX, softmax, average over views
  │
  ▼
[H] average per-crop softmax probabilities → argmax → disease label
  │
  ▼
return {detected, disease, confidence, boxes, processing_ms}
```

Each lettered step is documented below.

---

## 3. The detector — `detector_fold2.onnx` (and other folds)

### Input

| field | value |
|---|---|
| input name | `images` |
| shape | `[1, 3, 640, 640]` (**batch fixed at 1**) |
| dtype | `float32` |
| channel order | RGB |
| value range | `[0.0, 1.0]` (i.e. divide pixel byte by 255) |
| layout | NCHW (channels-first) |
| **no** ImageNet mean/std normalization | this is YOLOv8 — only `/255` |

### Preprocessing (step A)

The detector input is fixed at 640×640. To preserve aspect ratio you
must **letterbox** the image — scale the longer side to 640, pad the
shorter side with a constant color (114, 114, 114 in RGB is YOLOv8's
convention) to make the canvas square. Record the scale factor `r` and
the `(pad_x, pad_y)` offsets so you can invert the mapping in step D.

Letterbox recipe (conceptually):

1. `r = 640 / max(orig_w, orig_h)`
2. `new_w = round(orig_w * r)`, `new_h = round(orig_h * r)`
3. Resize the image (bilinear) to `(new_w, new_h)`.
4. Compute padding: `pad_x = (640 - new_w) / 2`, `pad_y = (640 - new_h) / 2`.
5. Paste the resized image onto a 640×640 canvas filled with `(114,114,114)`,
   at offset `(pad_x, pad_y)`.
6. Convert BGR→RGB if your image library decoded as BGR.
7. Cast to float32, divide by 255.
8. Transpose HWC → CHW, add a batch dim → final shape `[1,3,640,640]`.

### Output

| field | value |
|---|---|
| output name | `output0` |
| shape | `[1, 5, 8400]` |
| dtype | `float32` |

The detector was trained as a **binary** detector — every disease class
was collapsed to a single `lesion` class. That is why the channel axis
is 5 and not `4 + n_classes`. The 5 channels are:

| index | meaning |
|---|---|
| 0 | `cx` — box center x, in **letterboxed 640×640 pixel coordinates** |
| 1 | `cy` — box center y, in letterboxed 640×640 pixel coordinates |
| 2 | `w` — box width in letterboxed pixels |
| 3 | `h` — box height in letterboxed pixels |
| 4 | `conf` — sigmoid of objectness × class (in `[0,1]`) — already a probability, **do not sigmoid again** |

There are 8400 anchor predictions per image (YOLOv8's standard for
640×640 at strides 8/16/32: 80² + 40² + 20² = 8400).

> Note the axis order: channels-first, anchors-last. Many post-processing
> snippets you'll find online expect `[1, 8400, 5]`; transpose first if
> your code is structured that way.

### Decoding (step B) and NMS (step C)

For each of the 8400 anchors:

1. Read `(cx, cy, w, h, conf)`.
2. Skip if `conf < detector_conf` (production default `0.10`).
3. Convert center-form to corner-form (still in letterboxed pixels):
   - `x1 = cx - w/2`, `y1 = cy - h/2`, `x2 = cx + w/2`, `y2 = cy + h/2`.
4. Clip to `[0, 640]`.

Then run **Non-Maximum Suppression**:

- Sort surviving boxes by `conf` descending.
- IoU threshold: `0.45` (YOLOv8 default; suitable here — there is only
  one class, so no per-class NMS needed).
- Standard greedy NMS: repeatedly pick the highest-confidence box, drop
  any remaining box with IoU above the threshold against it.

A clean ONNX-runtime ecosystem typically has a helper for this
(`onnxruntime-extensions`, `yolov8` JS packages, etc.). If you write it
yourself it's ~20 lines.

### Unletterboxing (step D)

The surviving boxes are in **letterboxed 640×640 space**. Map them back
to the original image coordinates using the `r`, `pad_x`, `pad_y` you
recorded in step A:

- `orig_x = (letterbox_x - pad_x) / r`
- `orig_y = (letterbox_y - pad_y) / r`

Clip the final corners to `[0, orig_w] × [0, orig_h]`.

These coordinates are what you will return to the frontend in the
`boxes` array, **and** what you will use to crop in step F.

### What if no boxes survive?

Return immediately with `{detected: false, disease: null, confidence: null, boxes: [], processing_ms: …}`. Do not run the classifier.

---

## 4. Box merge (step E) — optional, off by default

If `merge=true` (dev-panel option, not used in production), cluster
overlapping boxes before classification:

- Greedy union-rect clustering: take boxes in confidence order, and for
  each new box assign it to the existing cluster whose union-rect has
  the highest IoU with this box, provided that IoU ≥ `0.3`. Otherwise
  start a new cluster.
- The merged box for each cluster is the bounding rectangle that
  encloses all its members; the merged confidence is the **max** of the
  members.

Production runs with `merge=false`. See [box_ops.py](box_ops.py) for an
exact reference.

---

## 5. Cropping (step F)

For each surviving box `(x1, y1, x2, y2)` in **original-image coordinates**:

1. Compute box width `w = x2 - x1` and height `h = y2 - y1`.
2. Pad the box by `P × w` horizontally and `P × h` vertically:
   - `nx1 = max(0,         round(x1 - P*w))`
   - `ny1 = max(0,         round(y1 - P*h))`
   - `nx2 = min(orig_w,    round(x2 + P*w))`
   - `ny2 = min(orig_h,    round(y2 + P*h))`
3. If `nx2 ≤ nx1` or `ny2 ≤ ny1`, drop the box.
4. Crop the original image to `[ny1:ny2, nx1:nx2]`.

`P` is the padding fraction. **Production default `P = 0.20`.** The
classifier was trained on crops padded by `0.40` and we deliberately
serve at `0.20` — sweeping showed `0.20` wins on conditional accuracy
by a small margin. Do not change this without re-running the eval.

See [crops.py](crops.py) for an exact reference (this geometry must
stay byte-equivalent to training).

If every box is dropped after cropping, return `{detected: false, …}`
as in step C.

---

## 6. The classifier — `classifier.onnx`

### Input

| field | value |
|---|---|
| input name | `input` |
| shape | `[batch_size, 3, 224, 224]` (**batch is dynamic** — you may stack TTA views and/or multiple crops) |
| dtype | `float32` |
| channel order | RGB |
| layout | NCHW |
| normalization | ImageNet mean/std (see below) |

Per-channel normalization values:

```
mean = [0.485, 0.456, 0.406]   # R, G, B
std  = [0.229, 0.224, 0.225]   # R, G, B
```

So each pixel is: `(pixel/255 - mean) / std`, applied independently per
channel.

### Preprocessing (step G)

For each crop produced in step F:

1. Convert to RGB if it is BGR.
2. Resize so that the **shorter** side is 224 (preserving aspect ratio,
   bilinear). This matches torchvision's `Resize(224)`.
3. **Center-crop** to 224×224. This matches torchvision's
   `CenterCrop(224)`.
4. Scale to `[0,1]` by dividing by 255.
5. ImageNet-normalize using the values above.
6. Transpose HWC → CHW.

### Test-time augmentation (TTA) — on by default

For each crop, also generate 3 augmented views and average their
softmax probabilities. The 4 views are:

1. Identity (the crop itself).
2. Horizontal flip.
3. Rotation by +10° (about the center).
4. Rotation by −10°.

Each view is preprocessed independently per step G. You can stack all
4 views into a single batch of shape `[4, 3, 224, 224]` and call the
classifier once, then average softmax across the 4 outputs.

If `tta=false`, use only view 1.

### Output

| field | value |
|---|---|
| output name | `output` |
| shape | `[batch_size, 5]` |
| dtype | `float32` |
| meaning | raw logits — **you must apply softmax yourself** |

Apply softmax along the class axis (axis 1) to get per-class
probabilities in `[0,1]` summing to 1.

### Class index → label

The output index maps to disease names in this **fixed order**:

| index | id (`disease`) | display (`disease_display`) |
|---|---|---|
| 0 | `Leukoplakia` | "Leukoplakia" |
| 1 | `Erythroplakia` | "Erythroplakia" |
| 2 | `OSMF` | "Oral Submucous Fibrosis" |
| 3 | `Lichen_Planus` | "Lichen Planus" |
| 4 | `NH_Ulcers` | "Non-healing Ulcer" |

There is **no "Normal" class**. Healthy = the detector returned no
boxes (handled in step C). The classifier is only ever asked to choose
between these 5 diseases, conditional on the detector having fired.

The `disease` value is the machine-readable id used by the API
contract; `disease_display` is the patient-facing label. Both come
from this same table — see [config.py](config.py) `CLASS_NAMES` and
`DISPLAY_NAMES` for the authoritative source.

---

## 7. Aggregating across crops and views (step H)

You will end up with one softmax probability vector per (crop, view)
pair. Aggregate in this order:

1. **TTA fold**: average the softmax vectors across the (up to 4) views
   of the same crop → one vector per crop.
2. **Box fold**: average the per-crop vectors across all crops in the
   image → one vector per image.
3. **Argmax** that final vector to pick the disease index.
4. The `confidence` reported to the user is the **value at that
   argmax index** — the predicted class's probability after averaging.

Always average **post-softmax** (not logits). Logit-space averaging is
not equivalent and the model was tuned against post-softmax aggregation.

---

## 8. The response shape

The reference service returns:

```json
{
  "detected": true,
  "disease": "Lichen_Planus",
  "disease_display": "Lichen Planus",
  "confidence": 0.74,
  "boxes": [
    {"x1": 412.5, "y1": 198.2, "x2": 803.7, "y2": 561.4, "conf": 0.83}
  ],
  "processing_ms": 412
}
```

When the detector finds nothing, all classifier-related fields are null:

```json
{
  "detected": false,
  "disease": null,
  "disease_display": null,
  "confidence": null,
  "boxes": [],
  "processing_ms": 87
}
```

`boxes` coordinates are in **original image pixels** (post-unletterbox,
pre-padding — i.e. the raw detector boxes mapped back, *not* the
padded crop windows).

---

## 9. Production configuration (MVP)

This is the minimum you need to ship a backend that matches the
reference service. Use these values verbatim:

| step | parameter | value |
|---|---|---|
| detector model | weight file | `weights/detector_fold2.onnx` |
| detector | input size | 640×640 |
| detector | preprocessing | letterbox (114 fill), RGB, `/255`, no mean/std |
| C — filter | `detector_conf` | **0.10** |
| C — NMS | IoU threshold | 0.45 |
| E — merge | enabled | **false** |
| F — crop | `padding` (`P`) | **0.20** |
| classifier | weight file | `weights/classifier.onnx` |
| classifier | input size | 224×224 |
| classifier | preprocessing | resize-shorter-side-to-224, center-crop, `/255`, ImageNet mean/std |
| G — TTA | enabled | **true** (4 views: identity, hflip, rot +10°, rot −10°) |
| H — aggregation | order | post-softmax mean over TTA, then over crops, then argmax |

That is the entire MVP. Build this first, get it matching the
reference outputs on the sample images in [examples/](examples/), then
optionally expose the dev-panel knobs in §10.

---

## 10. Dev-panel settings (optional, post-MVP)

The reference service exposes a "dev panel" so internal users can probe
the model. None of these are required for end-users — keep them behind
an internal flag or only enable them on staging.

### detector_fold ∈ {0, 2, 3, 4}

Swap which detector weight file is loaded. Each fold was trained on a
different 80/20 split of the dataset, so they make different mistakes.
The frontend labels them:

| fold | label | character |
|---|---|---|
| 0 | "Conservative" | trained at a less aggressive operating point; fires least often, fewest false alarms, lowest catch rate |
| 2 | **default** | best balance: ~97% catch, ~4% false-alarm at `conf=0.10` |
| 3 | "Alternative" | similar to fold 2, slightly more aggressive |
| 4 | "Aggressive" | highest catch, also most false alarms |

These are drop-in replacements at the ONNX level — same input/output
contract as fold 2. Keep one ONNX session per fold loaded (lazy-load
on first use to save memory if you don't need all four).

### detector_conf ∈ {0.01, 0.05, 0.10, 0.15, 0.25, 0.40, 0.50}

Filter threshold in step C. Lower = catches more lesions but flags
more healthy mouths. Production is **0.10**. The allowed values are
restricted to this discrete set so the user can't pick e.g. `0.123`
that we never benchmarked.

### padding ∈ {0.0, 0.2, 0.4}

The `P` in step F. Production is **0.20**. Going below 0.0 is
forbidden; above 0.4 is forbidden — both extremes hurt accuracy in
our sweeps.

### tta ∈ {true, false}

Toggle TTA averaging in step G. Off = single-view, ~3× faster but
~3 percentage points lower conditional accuracy. Production is **on**.

### merge ∈ {true, false}

Toggle step E. Production is **off**. Null result in combination with
TTA — they reduce the same kind of variance.

### Validating the allowed values

If you expose these to a user-facing settings UI, reject any value
outside the allowed sets above with a 400. The reference service's
validation is in [config.py](config.py) (`ALLOWED_FOLDS`,
`ALLOWED_CONFS`, `ALLOWED_PADDINGS`).

---

## 11. Performance notes

- Load each ONNX session **once** at process startup (or lazy-load on
  first use) and keep it warm. Cold-loading a 17 MB classifier on
  every request will dominate latency.
- Detector batch is fixed at 1 — you cannot batch images across
  requests through the detector ONNX. The classifier batch is dynamic;
  exploit this to send all (crops × TTA views) of one image in a
  single call.
- The reference Python service typically runs end-to-end in
  100–500 ms on CPU per image, depending on image size and how many
  boxes the detector returns. GPU is not required.
- The pipeline is read-only per request — safe to serve concurrent
  requests against the same ONNX session, subject to your runtime's
  threading model.

---

## 12. Sanity-checking your implementation

Before declaring done, run your backend against the images in
[examples/](examples/) and compare to the reference Python service's
output. The match should be **exact** on `disease` and within
~0.01 on `confidence`. If you see differences larger than that,
likely culprits, in priority order:

1. Channel order swapped somewhere (BGR vs RGB) — produces wildly
   wrong predictions.
2. Wrong normalization (e.g. `/255` for the classifier but no ImageNet
   mean/std, or vice versa for the detector) — also wildly wrong.
3. Letterbox padding color wrong, or aspect ratio not preserved — moves
   boxes by tens of pixels.
4. Resize using nearest-neighbour instead of bilinear — small but
   measurable accuracy hit on the classifier.
5. Resize→center-crop done in the wrong order (e.g. resize directly
   to 224×224 instead of resize-shorter-side-then-center-crop) — small
   but real accuracy hit and a tiny shift in confidence.
6. TTA rotation in degrees vs radians, or wrong sign on the angles —
   confidence drifts but argmax usually survives.
7. Averaging logits instead of softmax probabilities — confidence is
   miscalibrated even when argmax is correct.

If you have access to a Python environment, [inference.py](inference.py)
is the reference implementation; the values it produces are the ground
truth your backend should reproduce.
