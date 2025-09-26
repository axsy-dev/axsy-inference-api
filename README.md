Axsy Inference API
===================

FastAPI service exposing YOLO detection and CNN classification on images. Detection defaults to local `axsy-yolo.pt`; classification defaults to local `axsy-classifier.pt`.

Run locally
-----------

1) Install deps

```bash
pip install -r requirements.txt  # or: pip install fastapi uvicorn pillow ultralytics google-cloud-storage
```

2) Start API

```bash
# If using GCS, set credentials before starting the server
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json

uvicorn server:get_app --factory --host 0.0.0.0 --port 3000
```

3) Test detection (defaults to `axsy-yolo.pt`):

```bash
curl -sS -X POST http://localhost:3000/infer \
  -F "image=@/path/to/image.jpg"
```
4) Test classification (defaults to `axsy-classifier.pt`):

```bash
curl -sS -X POST http://localhost:3000/classify \
  -F "image=@/path/to/image.jpg"
```


Headers
-------
- `detector` (optional): absolute path or `gs://` path to YOLO model. If omitted, defaults to `./axsy-yolo.pt`.
- `gcs_bucket` (optional): bucket name when `detector` is a blob path (no scheme).
- `customer_id`, `model_id` (optional): when both are provided and `detector` is a blob path, the bucket is inferred as `customer_id`, and the blob as `model_id + '/' + detector`.

Classification headers
----------------------
- `classifier` (optional): absolute path or `gs://` path to classifier weights. If omitted, defaults to `./axsy-classifier.pt`.
- `gcs_bucket`, `customer_id`, `model_id` behave the same as detection for resolving remote paths.

Classification response
-----------------------
```json
{
  "classifier": "...",
  "result": {
    "input_size": 32,
    "top_index": 12,
    "top_prob": 0.93,
    "probs": [0.01, 0.00, ...]
  }
}
```

Response
--------
```json
{
  "detector": "...",
  "result": {
    "image": {"width": 1928, "height": 2472},
    "speed_ms": {"preprocess": 2.7, "inference": 454.0, "postprocess": 8.4},
    "num_detections": 74,
    "class_counts": {"shelf": 11, "product": 63},
    "detections": [
      {
        "class_id": 2,
        "class_name": "shelf",
        "confidence": 0.9554,
        "box": {
          "xyxy": [x1, y1, x2, y2],
          "center_xywh": [cx, cy, w, h],
          "center_xywh_norm": [cxn, cyn, wn, hn]
        }
      }
    ]
  }
}
```

Operational notes
-----------------
- The API is multi-user safe: model downloads are serialized per file, models are cached, and inference runs under a per‑model lock in a threadpool.
- Keep `smart-vision-trainiing-sa.json` and large model weights out of git.

