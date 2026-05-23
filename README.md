# dentilligence-api

FastAPI service wrapping the OPMD detection pipeline (YOLOv8n detector →
EfficientNet-B0 5-class classifier with 4-view TTA), plus a single-file
HTML frontend that drops into a WordPress page.

See [`app_instructions.md`](app_instructions.md) for the full build brief
and pipeline contract. See [`MODEL_SUMMARY.md`](MODEL_SUMMARY.md) for the
architecture, what the weights are, and the accuracy numbers.

---

## What's in this repo

```
dentilligence-api/
├── main.py, inference.py, models.py, ...   ← FastAPI service
├── weights/                                ← model weights (~41 MB total, in git)
├── frontend/index.html                     ← the WordPress page
├── Dockerfile, requirements.txt            ← deployment
└── tests/                                  ← smoke tests + 3 sample images
```

The service has one job: accept a mouth photo, return JSON with the
predicted disease (or "no lesion found") and bounding boxes. The frontend
talks to it from the user's browser.

---

## Local run (for development on a laptop)

Requires Python 3.10 or 3.11 and ~2 GB free RAM.

```bash
pip install -r requirements.txt
uvicorn main:app --port 8080
```

In a second terminal:
```bash
cd frontend
python -m http.server 8000
```

Open <http://localhost:8000> in a browser. The frontend auto-detects
`localhost` and points to the local API at `http://localhost:8080`.

Quick API checks:
```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@tests/samples/leuko.jpg"
```

---

# Deployment — Part 1 of 2: API on Google Cloud Run

This is for the GCP engineer. Total time: ~15 minutes (most of it waiting
for the Docker build).

## What you need before you start

1. A Google Cloud account with billing enabled.
2. A GCP project — create one at <https://console.cloud.google.com/projectcreate>
   if you don't already have one. **Write down the Project ID** (not the
   name) — you'll paste it into commands below.
3. The `gcloud` CLI installed on your machine —
   <https://cloud.google.com/sdk/docs/install>.
4. This repo cloned locally:
   ```bash
   git clone <repo-url>
   cd dentilligence-api
   ```

## Step 1 — One-time GCP setup

Replace `YOUR_PROJECT_ID` with the Project ID you wrote down.

```bash
# Log in (opens a browser tab)
gcloud auth login

# Tell gcloud which project to work in
gcloud config set project YOUR_PROJECT_ID

# Turn on the two APIs we use: Cloud Run (to host the service) and
# Cloud Build (to build the Docker image). First run takes ~30s.
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

If `gcloud auth login` complains about application-default credentials,
also run:
```bash
gcloud auth application-default login
```

## Step 2 — Update the CORS allowlist before you deploy

The API will refuse browser requests from origins it doesn't know.
Edit [`config.py`](config.py), find `WORDPRESS_ORIGIN`, and set it to
the exact origin where the WordPress page will be served, e.g.:

```python
WORDPRESS_ORIGIN = "https://dentilligence.com"
```

The origin is the protocol + domain — **no trailing slash, no path**.
If the page lives at `https://dentilligence.com/poc/opmd`, the origin
is still `https://dentilligence.com`.

If you have more than one origin (e.g. staging + prod), add them to the
`CORS_ORIGINS` list in the same file.

## Step 3 — Build the Docker image

This runs the build in Google's cloud (no local Docker needed). It will
upload the repo, build the image, and push it to Google Container Registry.
First build takes ~5–8 minutes; later builds are faster because of
caching.

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/dentilligence-api
```

You'll see live build output. The final line should say `SUCCESS`. If it
fails, the most common cause is forgetting to enable the Cloud Build API
in Step 1.

## Step 4 — Deploy to Cloud Run

```bash
gcloud run deploy dentilligence-api \
  --image gcr.io/YOUR_PROJECT_ID/dentilligence-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --timeout 60 \
  --max-instances 5
```

After ~1 minute it prints a URL like:
```
Service URL: https://dentilligence-api-xxxxxxx-uc.a.run.app
```

**Copy that URL — you need it for the WordPress step.**

Flag-by-flag explanation:
- `--region us-central1` — pick a region close to your users.
  Other options: `asia-south1` (Mumbai), `europe-west1` (Belgium),
  `asia-southeast1` (Singapore). You can change this; the URL changes too.
- `--allow-unauthenticated` — anyone with the URL can call the API.
  Required so the WordPress page can call it without an API key.
- `--memory 4Gi` — Torch + the model weights need ~2 GB resident; we
  give 4 GB to leave headroom for concurrent requests.
- `--cpu 2` — inference is CPU-bound; 2 cores cuts each request's
  wall-clock time roughly in half.
- `--timeout 60` — a request takes 0.5–2 s normally; 60 s is generous
  cover for cold starts.
- `--max-instances 5` — a hard cost cap. Cloud Run will spawn at most 5
  parallel instances; if traffic exceeds that, requests queue. Raise
  this number if you start seeing queueing in the Cloud Run dashboard.

## Step 5 — Smoke-test the deployed API

```bash
# Should print {"status":"ok",...}
curl https://YOUR-CLOUD-RUN-URL/health

# Should print a JSON result with detected/disease/boxes
curl -X POST https://YOUR-CLOUD-RUN-URL/predict \
  -F "image=@tests/samples/leuko.jpg"
```

The very first call after a deploy is slow (~10–20 s) because Cloud Run
is spinning up the container and loading model weights for the first
time. Subsequent calls are fast (<1 s).

## Re-deploying after a code change

```bash
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/dentilligence-api
gcloud run deploy dentilligence-api \
  --image gcr.io/YOUR_PROJECT_ID/dentilligence-api \
  --region us-central1
```
(The flags from the first deploy are remembered, so you don't need to
repeat them.)

## Costs — what to expect

Cloud Run charges per request and per second of CPU/RAM that the container
is actually handling a request. **It scales to zero when idle, so an
unused service costs $0/month.** At light traffic (a few hundred scans
per day) the bill is typically <$5/month. The biggest cost driver is
cold starts — each cold start uses ~10 s of CPU+RAM to load the model.

---

# Deployment — Part 2 of 2: Frontend on WordPress

This is for whoever maintains the Dentilligence/POC WordPress site. It
assumes Part 1 is done and you have the Cloud Run URL from Step 4.

## Step 1 — Edit the Cloud Run URL into the frontend

Open [`frontend/index.html`](frontend/index.html) in a text editor and
find this block near the top of the `<script>` section:

```js
var API_URL = (window.location.hostname === 'localhost'
               || window.location.hostname === '127.0.0.1')
  ? 'http://localhost:8080'
  : 'https://<cloud-run-url>';
```

Replace `https://<cloud-run-url>` with the actual Service URL from
Cloud Run, e.g.:

```js
  : 'https://dentilligence-api-xxxxxxx-uc.a.run.app';
```

**Don't add a trailing slash.** Save the file.

## Step 2 — Add the page in WordPress

1. Log into WordPress admin → **Pages → Add New**.
2. Give the page a title (e.g. "OPMD Screening").
3. In the block editor, click the **+** to add a new block, search for
   **"Custom HTML"**, and add it.
4. Open the edited `frontend/index.html` file in a text editor, select
   **all** of it (Ctrl/Cmd+A), copy, and paste it into the Custom HTML
   block.
5. Click **Preview** to verify it works (you'll see the phone mockup;
   try uploading a test photo).
6. Click **Publish** when satisfied.

## Step 3 — Verify the page works on the live site

Open the published page in a normal browser tab (not the WP editor
preview). Upload a test image. You should see:
- The "Running detector" chip, then bounding boxes, then a result card.
- Open browser devtools → **Network** tab → confirm `POST /predict`
  returns a 200 with JSON. If you see a **CORS error** (red), the
  page's origin doesn't match `WORDPRESS_ORIGIN` in `config.py` — go
  back to Part 1, Step 2, fix it, and re-deploy.

## Common WordPress gotchas

- **Some WordPress themes strip `<script>` tags from Custom HTML blocks**
  for security. If the page loads but nothing happens when you click
  Scan, this is the cause. Workarounds: (a) install the "HTML Snippets"
  or "Insert Headers and Footers" plugin and paste the HTML there, or
  (b) ask the site admin to allow `<script>` in Custom HTML for
  trusted editors.
- **Mixed content** — the WordPress page must be served over `https://`
  (it normally is). If it's `http://` it cannot call an `https://`
  Cloud Run URL; the browser blocks it.
- The HTML uses Google Fonts (Manrope, JetBrains Mono). If your WP
  site has a CSP that blocks external fonts, the page will still work,
  just with a fallback font.

---

## Updating the frontend later

Any change to `frontend/index.html` requires re-pasting the file into the
WordPress Custom HTML block. There is no automatic sync between this
repo and the WordPress site.

## Updating the API later

Re-run the two commands in Part 1, "Re-deploying after a code change".
The Cloud Run URL stays the same, so no WordPress change is needed.

---

# API reference

## `POST /predict`

Multipart form upload.

| field | type | default | allowed |
|---|---|---|---|
| `image` | file | required | jpeg, png, webp |
| `detector_fold` | int | `2` | `0`, `2`, `3`, `4` |
| `detector_conf` | float | `0.10` | `0.01`, `0.05`, `0.10`, `0.15`, `0.25`, `0.40`, `0.50` |
| `padding` | float | `0.20` | `0.0`, `0.2`, `0.4` |
| `tta` | bool | `true` | |
| `merge_boxes` | bool | `false` | |

Response on detection:
```json
{
  "detected": true,
  "disease": "OSMF",
  "disease_display": "Oral Submucous Fibrosis",
  "confidence": 0.78,
  "boxes": [{"x1": 120, "y1": 240, "x2": 380, "y2": 410, "conf": 0.42}],
  "processing_ms": 850
}
```

Response when no lesion is detected:
```json
{
  "detected": false,
  "disease": null,
  "disease_display": null,
  "confidence": null,
  "boxes": [],
  "processing_ms": 420
}
```

`boxes` are in original-image pixel coordinates.

## `GET /health`

```json
{"status": "ok", "model_loaded": true, "detector_folds_loaded": [2]}
```

Used by Cloud Run's health check and the frontend's "API up?" indicator.
