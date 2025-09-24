import io
import json
import os
from functools import lru_cache
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse
from PIL import Image
from ultralytics import YOLO
try:
    from google.cloud import storage
except Exception:
    storage = None  # Will validate at runtime when needed


app = FastAPI(title="Axsy Inference API")


def set_google_credentials_from_file(path: str) -> None:
    if path and os.path.isfile(path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = path


@lru_cache(maxsize=8)
def load_model_cached(model_path: str) -> YOLO:
    try:
        model = YOLO(model_path)
        return model
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {exc}")


def _ensure_storage_client(project_id: Optional[str] = None) -> "storage.Client":
    if storage is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "google-cloud-storage is not installed. Install it to load models from GCS."
            ),
        )
    try:
        # Always rely on credentials' default project to avoid header mismatch
        return storage.Client()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to init GCS client: {exc}")


def _download_blob_to_cache(bucket_name: str, blob_path: str, project_id: Optional[str] = None) -> str:
    client = _ensure_storage_client(None)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    if not blob.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Model object not found in GCS: gs://{bucket_name}/{blob_path}",
        )

    cache_dir = os.path.join(os.path.expanduser("~/.cache"), "axsy_inference", "models")
    os.makedirs(cache_dir, exist_ok=True)

    filename = os.path.basename(blob_path) or "model.pt"
    local_path = os.path.join(cache_dir, filename)

    # If file already cached, use it as-is
    if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
        return local_path

    blob.download_to_filename(local_path)
    return local_path


def resolve_model_path(
    classifier_value: str,
    gcs_bucket: Optional[str],
    customer_id: Optional[str] = None,
    model_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> str:
    # 1) Absolute local path
    if os.path.isabs(classifier_value) and os.path.isfile(classifier_value):
        return classifier_value

    # 2) Relative local path (resolve to cwd)
    rel_path = os.path.abspath(classifier_value)
    if os.path.isfile(rel_path):
        return rel_path

    # 3) GCS path explicit
    if classifier_value.startswith("gs://"):
        # gs://bucket/path/to/model.pt
        without_scheme = classifier_value[len("gs://") :]
        parts = without_scheme.split("/", 1)
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid gs:// path provided")
        bucket_name, blob_path = parts
        return _download_blob_to_cache(bucket_name, blob_path, None)

    # 4) GCS path via provided bucket header or env var
    bucket_name = gcs_bucket or os.getenv("GCS_BUCKET")
    if bucket_name:
        # Interpret classifier value as the blob path
        return _download_blob_to_cache(bucket_name, classifier_value, None)

    # 4b) Infer from customer_id/model_id headers: bucket = customer_id; blob = model_id + '/' + classifier
    if customer_id and model_id:
        inferred_blob = f"{model_id.rstrip('/')}/{classifier_value.lstrip('/')}"
        return _download_blob_to_cache(customer_id, inferred_blob, None)

    # If we reach here, we couldn't resolve the model
    raise HTTPException(
        status_code=400,
        detail=(
            "Classifier not found locally and no GCS bucket specified. "
            "Provide an absolute local path, a gs:// URL, or include 'gcs_bucket' header or GCS_BUCKET env var."
        ),
    )


def run_yolo_inference(model: YOLO, pil_image: Image.Image):
    results = model(pil_image)
    if not results:
        return {
            "image": {"width": None, "height": None},
            "speed_ms": {"preprocess": None, "inference": None, "postprocess": None},
            "num_detections": 0,
            "class_counts": {},
            "detections": [],
        }

    result = results[0]
    names = model.names

    def name_for(cls_id: int) -> str:
        if isinstance(names, dict):
            return names.get(cls_id, str(cls_id))
        return str(cls_id)

    # Image size (H, W)
    try:
        height, width = result.orig_shape  # (h, w)
    except Exception:
        width = getattr(pil_image, "width", None)
        height = getattr(pil_image, "height", None)

    # Speed metrics in ms (Ultralytics stores in result.speed)
    speed = getattr(result, "speed", {}) or {}
    speed_ms = {
        "preprocess": float(speed.get("preprocess", 0.0)) if isinstance(speed.get("preprocess", None), (int, float)) else None,
        "inference": float(speed.get("inference", 0.0)) if isinstance(speed.get("inference", None), (int, float)) else None,
        "postprocess": float(speed.get("postprocess", 0.0)) if isinstance(speed.get("postprocess", None), (int, float)) else None,
    }

    detections: list[Dict[str, Any]] = []
    class_counts: Dict[str, int] = {}
    if hasattr(result, "boxes") and result.boxes is not None:
        num_boxes = len(result.boxes)
        for i in range(num_boxes):
            cls_id = int(result.boxes.cls[i].item())
            conf = float(result.boxes.conf[i].item())
            x1, y1, x2, y2 = [float(v) for v in result.boxes.xyxy[i].tolist()]
            # Center xywh absolute and normalized
            center_xywh = [float(v) for v in result.boxes.xywh[i].tolist()]
            center_xywh_norm = [float(v) for v in result.boxes.xywhn[i].tolist()]

            class_name = name_for(cls_id)
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "box": {
                        "xyxy": [x1, y1, x2, y2],
                        "center_xywh": center_xywh,
                        "center_xywh_norm": center_xywh_norm,
                    },
                }
            )
    else:
        num_boxes = 0

    return {
        "image": {"width": width, "height": height},
        "speed_ms": speed_ms,
        "num_detections": int(num_boxes),
        "class_counts": class_counts,
        "detections": detections,
    }


@app.post("/infer")
async def infer(
    request: Request,
    file: Optional[UploadFile] = File(default=None, description="Image file to classify"),
    image: Optional[UploadFile] = File(default=None, description="Alternative field name for the image"),
    customer_id: Optional[str] = Header(default=None),
    model_id: Optional[str] = Header(default=None),
    project_id: Optional[str] = Header(default=None),
    classifier: Optional[str] = Header(default=None, description="Relative or absolute model path"),
    gcs_bucket: Optional[str] = Header(default=None, description="GCS bucket containing the classifier when classifier is a blob path"),
):
    # Ensure Google creds are available
    creds_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "smart-vision-trainiing-sa.json"))
    set_google_credentials_from_file(creds_path)

    # Fallback to header variants if not provided via explicit Header params
    headers = request.headers
    customer_id = customer_id or headers.get("customer_id") or headers.get("customer-id")
    model_id = model_id or headers.get("model_id") or headers.get("model-id")
    project_id = project_id or headers.get("project_id") or headers.get("project-id")
    classifier = classifier or headers.get("classifier") or headers.get("model")
    # Support both hyphen/underscore for gcs bucket header
    gcs_bucket = gcs_bucket or headers.get("gcs_bucket") or headers.get("gcs-bucket") or os.getenv("GCS_BUCKET")

    if classifier is None:
        # Default to local detection model in project root
        default_model = os.path.abspath(os.path.join(os.path.dirname(__file__), "axsy-yolo.pt"))
        if not os.path.isfile(default_model):
            raise HTTPException(status_code=400, detail="Missing required header: classifier (and default axsy-yolo.pt not found)")
        classifier = default_model

    # Resolve model path locally or via GCS
    model_path = resolve_model_path(
        classifier,
        gcs_bucket,
        customer_id=customer_id,
        model_id=model_id,
        project_id=project_id,
    )

    # Load model (cached)
    model = load_model_cached(model_path)

    # Read file into PIL image (support both 'file' and 'image' fields)
    selected = file if file is not None else image
    if selected is None:
        raise HTTPException(status_code=400, detail="Missing image file. Use form field 'file' or 'image'.")
    try:
        contents = await selected.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    inference = run_yolo_inference(model, pil_image)

    response: Dict[str, Any] = {
        "customer_id": customer_id,
        "model_id": model_id,
        "project_id": project_id,
        "classifier": classifier,
        "result": inference,
    }

    return JSONResponse(content=response)


def get_app():
    return app


