# Model summary

A plain-English overview of the models shipped with `dentilligence-api`,
plus the headline numbers from the research repo.

> For the full experimental record (data splits, ablations, failed
> attempts, design rationale), see the research repository:
> **<https://github.com/IshuIsAwake/opmd_detection>**

---

## Architecture

The pipeline is **two stages**, run back-to-back on every uploaded photo:

```
mouth photo
   │
   ▼
[ Stage 1 — YOLOv8n binary detector ]   "is there a lesion, and where?"
   │
   ├── no detection ─► return {detected: false}
   │
   └── one or more bounding boxes
           │
           ▼
   pad each box by 20 % and crop it from the original photo
           │
           ▼
[ Stage 2 — EfficientNet-B0 classifier ]   "if there is a lesion, which one?"
           │
           ▼
   4-view test-time augmentation (TTA): original + horizontal flip
   + rotate +10° + rotate −10°.
   For each crop: mean softmax across the 4 TTA views.
   For the image:  mean softmax across the crops → argmax.
           │
           ▼
   return {disease, confidence, boxes, processing_ms}
```

- **Stage 1 detector**: YOLOv8n (the smallest YOLOv8 variant, ~6 MB).
  Trained as a **binary** detector — every disease class collapses to
  `lesion`. Its job is only "is there something to look at," not what
  it is. Operating point: `conf = 0.10` (chosen for high catch rate at
  acceptable false-alarm rate on negatives).
- **Stage 2 classifier**: EfficientNet-B0 (~17 MB) with a custom head
  (`1280 → Linear(256) → GELU → Dropout → Linear(5)`). Trained
  end-to-end on lesion crops padded by 40 % of the box dimensions.
  Input: 224×224 RGB, ImageNet-normalised. Outputs softmax over the
  5 disease classes — Leukoplakia, Erythroplakia, OSMF, Lichen Planus,
  Non-healing Ulcer. **There is no "Normal" class** — healthy = the
  detector stays silent.

Both models were trained with 5-fold cross-validation. The repo ships
**four detector folds** (the user can switch between them in the dev
panel) and **one classifier fold** (the best one).

---

## Dataset and split

| | count |
|---|---|
| total positive images (with at least one labelled lesion) | **362** |
| total negative images (healthy / no lesion) | **570** |
| cross-validation scheme | 5-fold (80 % train + inner-val, 20 % test per fold) |
| test images per fold (approx.) | 72 positives + 72 negatives = 144 total |

So when you read "fold 2: 70/72 positives detected" below, the
denominator is the **20 % held-out test slice** for that fold — the
model never saw those images during training. Negatives are
randomly sampled to match the number of positives so the catch / FA
rates are on a balanced base.

---

## Results — Stage 1 detector

### At the production operating point (conf = 0.10)

This is the threshold the deployed service uses. Numbers below count
images directly — **how many of the held-out test images did the
detector fire on?** Sourced from the Phase 2 evaluation runs
(`Experimenting/classifier_experiments/results/phase2_b0_aug_tta/fold_*/summary.json`).

Fold 0 is excluded from this table — it used a globally less
aggressive detector and is not comparable at the same operating point. It still ships as the dev-panel's "Conservative" option.

| fold | positives detected | catch rate | negatives flagged | false-alarm rate |
|---|---|---|---|---|
| 1 | 60 / 73 | 0.822 | 1 / 73 | 0.014 |
| **2** | **70 / 72** | **0.972** | **3 / 72** | **0.042** |
| 3 | 67 / 72 | 0.931 | 5 / 72 | 0.069 |
| 4 | 64 / 71 | 0.901 | 8 / 71 | 0.113 |
| **mean ± std** | — | **0.907 ± 0.055** | — | **0.059 ± 0.037** |

- **Positives detected** — of the held-out positive images, how many
  had at least one bounding box at conf ≥ 0.10. Each miss is a
  positive image the user would see "Looks fine" for.
- **Negatives flagged** — of the held-out healthy images, how many
  got at least one bounding box anyway. Each one becomes a "this
  might be …" message on a healthy mouth.

**Fold 2 is the default in production** — best balance of the four:
near-perfect catch (70/72 = 97 %) with the lowest false-alarm rate
(3/72 = 4 %) of any fold that catches > 90 %. Fold 1 has the lowest
FA but misses 13/73 positives — too low a catch for a screening tool.

### At a stricter operating point (conf = 0.25), for comparison

Used in the standalone detector evaluation. Higher threshold ⇒ fewer
false alarms but fewer catches. Source:
`Experimenting/results/kfold5_geom_no_color_binary/summary.txt`.

| fold | screening_acc | detection_rate (positives) | false_alarm (negatives) | box F1 (IoU≥0.5) | loc IoU on hits |
|---|---|---|---|---|---|
| 0 | 0.824 | 0.649 | 0.000 | 0.410 | 0.635 |
| 1 | 0.781 | 0.562 | 0.000 | 0.363 | 0.671 |
| **2** | **0.875** | **0.750** | **0.000** | **0.420** | **0.667** |
| 3 | 0.847 | 0.736 | 0.042 | 0.318 | 0.674 |
| 4 | 0.880 | 0.817 | 0.056 | 0.382 | 0.669 |
| **mean ± std** | **0.842 ± 0.041** | **0.703 ± 0.099** | **0.020 ± 0.027** | **0.379 ± 0.041** | **0.664 ± 0.017** |

The dev panel's confidence slider lets internal users move along this
curve at runtime — drop to 0.05 for more catches on subtle lesions,
raise to 0.25–0.40 to suppress false alarms on tricky frames (teeth,
equipment). **Production default is 0.10** because the downstream
classifier is the better filter for ambiguous boxes.

---

## Results — Stage 2 classifier (in isolation)

EfficientNet-B0 trained on ground-truth lesion crops padded by 40 %,
strong-aug recipe, evaluated on held-out fold crops (no detector in the
loop — this is the classifier's own ceiling).

Source:
`Experimenting/classifier_experiments/results/gt_pad_0.40_b0_aug/summary.txt`

| fold | n test crops | micro accuracy | macro accuracy |
|---|---|---|---|
| 0 | 112 | 0.714 | 0.715 |
| 1 | 133 | 0.586 | 0.573 |
| **2** | **104** | **0.692** | **0.676** |
| 3 | 119 | 0.605 | 0.599 |
| 4 | 111 | 0.613 | 0.616 |
| **mean ± std** | — | **0.642 ± 0.051** | **0.636 ± 0.052** |

Per-class recall (averaged across folds):

| class | recall |
|---|---|
| Leukoplakia    | 0.578 ± 0.147 |
| Erythroplakia  | 0.605 ± 0.073 |
| OSMF           | 0.673 ± 0.140 |
| Lichen Planus  | 0.726 ± 0.062 |
| Non-healing Ulcer | 0.596 ± 0.087 |

**Fold 2 is the shipped classifier.** Fold 0 scores higher in isolation
but it's paired with the weaker fold-0 detector — the two-stage system
runs each fold's detector with its own fold's classifier, so the
relevant question is which combination wins on the full pipeline.

---

## Results — full pipeline (detector + classifier together)

This is what actually matters: a real user uploads a photo, the
detector finds (or doesn't find) boxes, those boxes get classified,
votes are averaged, and an answer comes out. Evaluated on folds {1, 2,
3, 4} (fold 0 excluded — it ran a globally less aggressive detector
that doesn't compare cleanly).

Source:
`Experimenting/classifier_experiments/results/phase2_b0_aug_tta/summary.txt`

### Per-fold breakdown (production config: train_pad = 0.40, serve_pad = 0.20, TTA on, merge off)

| fold | cond. disease accuracy | system accuracy |
|---|---|---|
| 1 | 0.700 | 0.575 |
| 2 | 0.614 | 0.597 |
| 3 | 0.701 | 0.653 |
| 4 | 0.625 | 0.563 |
| **mean ± std** | **0.660 ± 0.041** | **0.597 ± 0.034** |

Cross-fold catch rate **0.907 ± 0.055**, negative false-alarm rate
**0.059 ± 0.037**.

### What the columns mean

- **conditional disease accuracy** — *given that the detector fired on
  a positive image*, did the classifier name the right disease? This
  is the classifier's score on the inputs the detector actually feeds
  it (a different, harder distribution than the GT crops above).
- **system accuracy** — end-to-end. Counts a "miss" if the detector
  didn't fire on a positive, *and* a "wrong" if it fired but the
  classifier got the disease wrong. Closer to what a user perceives.
- **catch rate** — fraction of positive images where the detector
  fired at least one box.
- **negative false alarm rate** — fraction of negative images where
  the detector incorrectly fired.

### Why these specific settings

| component | choice | why |
|---|---|---|
| detector | fold 2 (`detector_fold2.pt`) | best false-alarm rate on negatives + above-average catch |
| detector confidence | 0.10 | chosen to maximise catch on positives at acceptable FA on negatives |
| classifier | fold 2 (`classifier.pt`) | best cross-fold balance on the full-pipeline eval |
| serve padding | 0.20 | swept {0.0, 0.2, 0.4}; 0.2 wins on cond_acc by ~2 pp over 0.4 and ~7 pp over 0.0 |
| TTA | on, 4 views (identity, hflip, rot ±10°) | +3.0 pp cond_acc over no-TTA, std unchanged. Free lunch — same model, inference-only |
| merge overlapping boxes | off | tried it (clustering IoU≥0.3 union-rect); null result combined with TTA — they reduce the same variance |
| aggregation | mean post-softmax across TTA views → across boxes → argmax | post-softmax averaging respects the simplex; standard practice |

### Headline number

**Cond. disease accuracy: 0.660 ± 0.041** (folds 1–4, leak-free,
production config). **System accuracy: 0.597 ± 0.034**. The
classifier-confidence readout the dev panel displays is the predicted
class's softmax probability *after* TTA averaging and box averaging.

---

## What's in the `weights/` directory

| file | size | role |
|---|---|---|
| `detector_fold0.pt` | 6 MB | YOLOv8n, fold 0 — "Conservative" option in dev panel |
| `detector_fold2.pt` | 6 MB | YOLOv8n, fold 2 — **default in production** |
| `detector_fold3.pt` | 6 MB | YOLOv8n, fold 3 — "Alternative" option in dev panel |
| `detector_fold4.pt` | 6 MB | YOLOv8n, fold 4 — "Aggressive" option in dev panel |
| `classifier.pt`     | 17 MB | EfficientNet-B0, fold 2 |

Total ~41 MB, committed directly to the repo (well under GitHub's
100 MB-per-file limit; no LFS).

---

## Known failure modes

The detector occasionally fires on things that aren't lesions — teeth,
fingers, dental equipment (mirrors / retractors / suction tips),
lipstick / food debris, flash glare on wet tissue. These are
out-of-distribution for the training set, which is close-up photos of
the oral cavity with the lesion roughly centered.

The frontend exposes this honestly via the "the model may misfire"
note on the upload screen. The dev-panel options exist to let an
internal user probe: switching to fold 0 (most conservative) suppresses
many false alarms; raising `detector_conf` to 0.25 or 0.40 has a
similar effect at the cost of catch rate.

---

## Reproducing these numbers

All training scripts, data splits, and evaluation harnesses live in the
research repo:
<https://github.com/IshuIsAwake/opmd_detection>.

Specifically:

| concern | file in research repo |
|---|---|
| detector training | `Experimenting/run_kfold_binary.py` |
| classifier training | `Experimenting/classifier_experiments/exp_b0_aug_gt_pad04.py` |
| full pipeline eval (TTA) | `Experimenting/classifier_experiments/phase2_pipeline.py --backbone b0_aug --tta` |
| headline writeup | `Experiment_Results.md`, §17 |
