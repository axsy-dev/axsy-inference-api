Axsy Inference API
===================

FastAPI service exposing YOLO detection on images. Defaults to local `axsy-yolo.pt` and returns structured detections with image metadata and speed.

Run locally
-----------

1) Install deps

```bash
pip install -r requirements.txt  # or: pip install fastapi uvicorn pillow ultralytics google-cloud-storage
```

2) Start API

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json  # optional if not using GCS
uvicorn server:get_app --host 0.0.0.0 --port 3000
```

3) Test detection (defaults to `axsy-yolo.pt`):

```bash
curl -sS -X POST http://localhost:3000/infer \
  -F "image=@/path/to/image.jpg"
```

Headers
-------
- `classifier` (optional): absolute path or GCS path to YOLO model. If omitted, defaults to `./axsy-yolo.pt`.
- `customer_id`, `model_id` (optional): when provided with `classifier` as a relative blob path, the bucket is inferred as `customer_id`, blob as `model_id + '/' + classifier`.

Response
--------
```json
{
  "classifier": "...",
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

Notes
-----
- Keep `smart-vision-trainiing-sa.json` and large model weights out of git (see `.gitignore`).

# Axsy Detection

Package to run detection on a single image file or a directory of images.

## Build and Install the Python Package:

```bash
python build.py --install
```

## Run ##
usage: 
`python -m axsy_detection [--bucket_id BUCKET_ID] [--dataset_id DATASET_ID] [--detector DETECTOR] [--input_path INPUT_PATH] [--no_inference] [--output_filename OUTPUT_FILENAME]`


## Input Options

### `--path`
- **Type:** `str`
- **Required:** `True`
- **Description:** The full file path to the `axsy_notation_data` directory 

### `--dataset_id`
- **Type:** `str`
- **Default:** `"dataset-0"`
- **Description:** The identifier for the dataset to be used.

### `--detector`
- **Type:** `str`
- **Default:** `"yolo.pt"`
- **Description:** The filename of the detector model. Must be in the `label-assist` directory Defaults to `/yolo.pt`.

### `--input`
- **Type:** `str`
- **Default:** `""`
- **Description:** The path to a single image file or directory of images to be processed. Defaults to the entire `images` directory

### `--no_inference`
- **Type:** `boolean` (flag)
- **Description:** Use this flag to disable the inference step. When provided, the script will not run inference on the input images.

---

## Example Usage

```bash
python -m axsy_detection --path /path/to/axsy_notation_data --input image1.jpg --detector my_model.pt
```

```bash
python build.py --install && python -m axsy_detection --path /Users/josephhills/Documents/Axsy/axsy-notation-data
```

