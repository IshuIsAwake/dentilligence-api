#!/usr/bin/env bash
# Sample POST /predict call. Replace URL + image path.
curl -s -X POST "http://localhost:8080/predict" \
  -F "image=@/path/to/mouth_photo.jpg" \
  -F "detector_fold=2" \
  -F "detector_conf=0.10" \
  -F "padding=0.2" \
  -F "tta=true" \
  -F "merge_boxes=false" | python -m json.tool
