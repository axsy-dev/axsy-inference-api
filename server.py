import io
import json
import os
import threading
from functools import lru_cache
import logging
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, File, Header, HTTPException, UploadFile, Request, Query
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, HTMLResponse
from PIL import Image
from ultralytics import YOLO
try:
    from google.cloud import storage
except Exception:
    storage = None  # Will validate at runtime when needed


app = FastAPI(title="Axsy Inference API")

# Logger
logger = logging.getLogger("uvicorn.error")

# Global locks for concurrency safety
_global_lock = threading.Lock()
_download_locks: Dict[str, threading.Lock] = {}
_model_locks: Dict[str, threading.Lock] = {}


def _get_lock(lock_map: Dict[str, threading.Lock], key: str) -> threading.Lock:
    with _global_lock:
        lock = lock_map.get(key)
        if lock is None:
            lock = threading.Lock()
            lock_map[key] = lock
        return lock


@lru_cache(maxsize=8)
def load_model_cached(model_path: str) -> YOLO:
    try:
        model = YOLO(model_path)
        return model
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load model: {exc}")


# --- Classifier support ---
@lru_cache(maxsize=8)
def load_classifier_cached(model_path: str):
    try:
        from classifier import load_classifier  # local import to avoid hard dep when unused
        model = load_classifier(model_path)
        return model
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to load classifier: {exc}")


def _load_labels_from_json(labels_path: str) -> Optional[list[str]]:
    """Load class labels from a JSON file if present. Returns None on any error."""
    try:
        if not os.path.isfile(labels_path):
            return None
        with open(labels_path, "r") as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            return data
        if isinstance(data, dict):
            for key in ("labels", "classes", "names"):
                val = data.get(key)
                if isinstance(val, list) and all(isinstance(x, str) for x in val):
                    return val
            for val in data.values():
                if isinstance(val, list) and all(isinstance(x, str) for x in val):
                    return val
        return None
    except Exception:
        return None


def load_classifier_labels() -> Optional[list[str]]:
    """Load class labels from axsy-classifier.json in project root if present (no caching)."""
    labels_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "axsy-classifier.json"))
    return _load_labels_from_json(labels_path)


def load_labels_for_classifier(
    classifier_value: Optional[str],
    model_path: Optional[str],
    gcs_bucket: Optional[str],
    customer_id: Optional[str],
    model_id: Optional[str],
    project_id: Optional[str],
    metadata_value: Optional[str] = None,
) -> Optional[list[str]]:
    """Resolve and load labels for a given classifier.

    Preference order:
    1) If a metadata header was provided, resolve it (local or remote) and load labels from it.
    2) If a classifier header was provided, try adjacent metadata JSON candidates using the same resolution rules:
       '<base>.metadata.json', '<base>.meta.json', '<base>.json'.
    3) Try local candidates next to the resolved model_path.
    4) Fallback to project-root axsy-classifier.json.
    """
    # 1) Explicit metadata header
    try:
        if metadata_value:
            try:
                resolved_meta = resolve_model_path(
                    metadata_value,
                    gcs_bucket,
                    customer_id=customer_id,
                    model_id=model_id,
                    project_id=project_id,
                )
                labels = _load_labels_from_json(resolved_meta)
                if labels:
                    return labels
            except Exception:
                pass
    except Exception:
        pass

    # 2) Resolve via the original classifier value if provided
    try:
        if classifier_value:
            base = os.path.splitext(classifier_value)[0]
            for suffix in (".metadata.json", ".meta.json", ".json"):
                candidate = base + suffix
                try:
                    resolved = resolve_model_path(
                        candidate,
                        gcs_bucket,
                        customer_id=customer_id,
                        model_id=model_id,
                        project_id=project_id,
                    )
                    labels = _load_labels_from_json(resolved)
                    if labels:
                        return labels
                except Exception:
                    # Best-effort: ignore and try next candidate
                    pass
    except Exception:
        pass

    # 3) Attempt local file adjacent to resolved model_path
    try:
        if model_path:
            base = os.path.splitext(model_path)[0]
            for suffix in (".metadata.json", ".meta.json", ".json"):
                local_candidate = base + suffix
                labels = _load_labels_from_json(local_candidate)
                if labels:
                    return labels
    except Exception:
        pass

    # 4) Fallback
    return load_classifier_labels()


def _ensure_storage_client(project_id: Optional[str] = None) -> "storage.Client":
    if storage is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "google-cloud-storage is not installed. Install it to load models from GCS."
            ),
        )
    try:
        creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if creds_path:
            try:
                exists = os.path.isfile(creds_path)
            except Exception:
                exists = False
            logger.info("GCS: using GOOGLE_APPLICATION_CREDENTIALS=%s (exists=%s)", creds_path, exists)
            if exists:
                try:
                    # Prefer explicit JSON when present
                    return storage.Client.from_service_account_json(creds_path)
                except Exception as exc:
                    logger.exception("GCS: Failed to init client from JSON %s: %s; falling back to ADC", creds_path, exc)
            # If path is missing or failed to load, ensure env var does not force google.auth to try it
            logger.warning("GCS: credentials file path does not exist or failed, clearing env and using ADC: %s", creds_path)
            prev = os.environ.pop("GOOGLE_APPLICATION_CREDENTIALS", None)
            try:
                return storage.Client()
            finally:
                if prev is not None:
                    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = prev
        else:
            logger.info("GCS: GOOGLE_APPLICATION_CREDENTIALS not set; using default application credentials")
        # Always rely on credentials' default project to avoid header mismatch
        return storage.Client()
    except Exception as exc:
        logger.exception("GCS: Failed to init storage client: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to init GCS client: {exc}")


def _download_blob_to_cache(bucket_name: str, blob_path: str, project_id: Optional[str] = None) -> str:
    client = _ensure_storage_client(None)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    logger.info("GCS: resolving gs://%s/%s", bucket_name, blob_path)
    if not blob.exists():
        logger.error("GCS: object not found gs://%s/%s", bucket_name, blob_path)
        raise HTTPException(
            status_code=404,
            detail=f"Model object not found in GCS: gs://{bucket_name}/{blob_path}",
        )

    cache_dir = os.path.join(os.path.expanduser("~/.cache"), "axsy_inference", "models")
    os.makedirs(cache_dir, exist_ok=True)

    filename = os.path.basename(blob_path) or "model.pt"
    local_path = os.path.join(cache_dir, filename)

    # Serialize downloads per file to avoid races
    lock = _get_lock(_download_locks, local_path)
    with lock:
        if os.path.isfile(local_path) and os.path.getsize(local_path) > 0:
            logger.info("GCS: using cached file %s", local_path)
            return local_path
        try:
            blob.download_to_filename(local_path)
            logger.info("GCS: downloaded gs://%s/%s -> %s", bucket_name, blob_path, local_path)
            return local_path
        except Exception as exc:
            logger.exception("GCS: download failed for gs://%s/%s: %s", bucket_name, blob_path, exc)
            raise HTTPException(status_code=500, detail=f"Failed to download GCS object: {exc}")


def resolve_model_path(
    detector_value: str,
    gcs_bucket: Optional[str],
    customer_id: Optional[str] = None,
    model_id: Optional[str] = None,
    project_id: Optional[str] = None,
) -> str:
    # 1) Absolute local path
    if os.path.isabs(detector_value) and os.path.isfile(detector_value):
        return detector_value

    # 2) Relative local path (resolve to cwd)
    rel_path = os.path.abspath(detector_value)
    if os.path.isfile(rel_path):
        return rel_path

    # 3) GCS path explicit
    if detector_value.startswith("gs://"):
        # gs://bucket/path/to/model.pt
        without_scheme = detector_value[len("gs://") :]
        parts = without_scheme.split("/", 1)
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid gs:// path provided")
        bucket_name, blob_path = parts
        return _download_blob_to_cache(bucket_name, blob_path, None)

    # 4) GCS path via provided bucket header or env var
    bucket_name = gcs_bucket or os.getenv("GCS_BUCKET")
    if bucket_name:
        # Interpret detector value as the blob path
        return _download_blob_to_cache(bucket_name, detector_value, None)

    # 4b) Infer from customer_id/model_id headers: bucket = customer_id; blob = model_id + '/' + detector
    if customer_id and model_id:
        inferred_blob = f"{model_id.rstrip('/')}/{detector_value.lstrip('/')}"
        return _download_blob_to_cache(customer_id, inferred_blob, None)

    # If we reach here, we couldn't resolve the model
    raise HTTPException(
        status_code=400,
        detail=(
            "Detector not found locally and no GCS bucket specified. "
            "Provide an absolute local path, a gs:// URL, or include 'gcs_bucket' header or GCS_BUCKET env var."
        ),
    )


def _prepare_image_tensor(pil_image: Image.Image, size: int, device: str):
    # Lazy import heavy deps to avoid impacting default detection endpoint
    import numpy as np  # type: ignore
    import torch  # type: ignore

    img = pil_image.resize((size, size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    chw = np.transpose(arr, (2, 0, 1))
    bchw = np.expand_dims(chw, axis=0)
    non_blocking = (device == "cuda")
    x = torch.from_numpy(bchw).to(device, non_blocking=non_blocking)
    return x


def _prepare_image_tensor_batch(pil_images: list[Image.Image], size: int, device: str):
    # Lazy import heavy deps only when needed
    import numpy as np  # type: ignore
    import torch  # type: ignore

    arrays = []
    for img in pil_images:
        im = img.resize((size, size))
        arr = np.asarray(im, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        chw = np.transpose(arr, (2, 0, 1))
        arrays.append(chw)
    bchw = np.stack(arrays, axis=0)
    non_blocking = (device == "cuda")
    x = torch.from_numpy(bchw).to(device, non_blocking=non_blocking)
    return x

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


def _run_inference_threadsafe(model_path: str, model: YOLO, pil_image: Image.Image) -> Dict[str, Any]:
    """Run inference under a per-model lock to ensure thread-safety.

    This prevents concurrent model() calls on the same model instance which
    can be unsafe with some backends (e.g., GPU execution).
    """
    lock = _get_lock(_model_locks, model_path)
    with lock:
        return run_yolo_inference(model, pil_image)


def _run_classification_threadsafe(model_path: str, model, pil_image: Image.Image) -> Dict[str, Any]:
    lock = _get_lock(_model_locks, model_path)
    with lock:
        # Derive input size from model attribute
        try:
            inp_size = int(model.sizes[0].item())
        except Exception:
            inp_size = 32
        # Resolve device without importing torch globally
        try:
            device = next(model.parameters()).device.type
        except Exception:
            device = "cpu"
        from time import perf_counter
        t0 = perf_counter()
        x = _prepare_image_tensor(pil_image, inp_size, device)
        t1 = perf_counter()
        from classifier import classify_image_tensor  # local import to keep deps lazy
        result = classify_image_tensor(model, x)
        t2 = perf_counter()
        preprocess_ms = float((t1 - t0) * 1000.0)
        inference_ms = float((t2 - t1) * 1000.0)
        return {
            "input_size": inp_size,
            "speed_ms": {"preprocess": preprocess_ms, "inference": inference_ms, "postprocess": None},
            **result,
        }


def _clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0.0, min(x1, w))
    y1 = max(0.0, min(y1, h))
    x2 = max(0.0, min(x2, w))
    y2 = max(0.0, min(y2, h))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _reclassify_products(
    pil_image: Image.Image,
    detections: list[Dict[str, Any]],
    model_path: str,
    classifier_model,
    labels: Optional[list[str]],
) -> Dict[str, Any]:
    if not detections:
        return {"num": 0, "speed_ms": {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}}
    try:
        w, h = pil_image.width, pil_image.height
        # Determine classifier input size and device
        try:
            inp_size = int(classifier_model.sizes[0].item())
        except Exception:
            inp_size = 64
        try:
            device = next(classifier_model.parameters()).device.type
        except Exception:
            device = "cpu"

        from time import perf_counter
        total_pre_ms = 0.0
        total_inf_ms = 0.0
        total_post_ms = 0.0
        # Collect product detections and crops
        prod_indices: List[int] = []
        crops: List[Image.Image] = []
        for i, det in enumerate(detections):
            if str(det.get("class_name")) != "product":
                continue
            x1, y1, x2, y2 = det.get("box", {}).get("xyxy", [0, 0, 0, 0])
            x1, y1, x2, y2 = _clamp_box(float(x1), float(y1), float(x2), float(y2), float(w), float(h))
            crop = pil_image.crop((x1, y1, x2, y2)).convert("RGB")
            crops.append(crop)
            prod_indices.append(i)

        if not crops:
            return {"num": 0, "speed_ms": {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}}

        t0 = perf_counter()
        batch_x = _prepare_image_tensor_batch(crops, inp_size, device)
        t1 = perf_counter()
        from classifier import classify_image_batch  # lazy import
        out = classify_image_batch(classifier_model, batch_x)
        t2 = perf_counter()

        top_indices: List[int] = [int(v) for v in out.get("top_indices", [])]
        top_probs: List[float] = [float(v) for v in out.get("top_probs", [])]

        for offset, det_idx in enumerate(prod_indices):
            idx_val = int(top_indices[offset]) if offset < len(top_indices) else -1
            prob_val = float(top_probs[offset]) if offset < len(top_probs) else 0.0
            det = detections[det_idx]
            det["classifier_id"] = idx_val
            if labels and 0 <= idx_val < len(labels):
                det["class_name"] = labels[idx_val]
            elif labels:
                det["class_name"] = labels[0]
            det["confidence"] = float(prob_val if prob_val else det.get("confidence", 0.0))

        t3 = perf_counter()
        total_pre_ms = float((t1 - t0) * 1000.0)
        total_inf_ms = float((t2 - t1) * 1000.0)
        total_post_ms = float((t3 - t2) * 1000.0)
        count = len(prod_indices)
    except Exception as exc:
        logger.exception("Reclassify: batch classification failed: %s", exc)
        # Fail-open: leave detections as-is
        return {"num": 0, "speed_ms": {"preprocess": None, "inference": None, "postprocess": None}}
    return {
        "num": count,
        "speed_ms": {"preprocess": total_pre_ms, "inference": total_inf_ms, "postprocess": total_post_ms},
    }


@app.post("/infer")
async def infer(
    request: Request,
    file: Optional[UploadFile] = File(default=None, description="Image file to classify"),
    image: Optional[UploadFile] = File(default=None, description="Alternative field name for the image"),
    customer_id: Optional[str] = Header(default=None),
    model_id: Optional[str] = Header(default=None),
    project_id: Optional[str] = Header(default=None),
    detector: Optional[str] = Header(default=None, description="Relative or absolute model path"),
    gcs_bucket: Optional[str] = Header(default=None, description="GCS bucket containing the detector when detector is a blob path"),
    reclassify_products: Optional[bool] = Header(default=None, description="If true, reclassify 'product' detections with classifier"),
    classifier: Optional[str] = Header(default=None, description="Relative or absolute classifier path for reclassification"),
    metadata: Optional[str] = Header(default=None, description="Optional path to metadata JSON containing labels for classifier reclassification"),
):
    # Fallback to header variants if not provided via explicit Header params
    headers = request.headers
    customer_id = customer_id or headers.get("customer_id") or headers.get("customer-id")
    model_id = model_id or headers.get("model_id") or headers.get("model-id")
    project_id = project_id or headers.get("project_id") or headers.get("project-id")
    detector = detector or headers.get("detector") or headers.get("model")
    # Support both hyphen/underscore for gcs bucket header
    gcs_bucket = gcs_bucket or headers.get("gcs_bucket") or headers.get("gcs-bucket") or os.getenv("GCS_BUCKET")
    # Optional classifier/metadata headers for reclassification
    classifier = classifier or headers.get("classifier") or headers.get("model")
    metadata = metadata or headers.get("metadata") or headers.get("meta")

    if detector is None:
        # Default to local detection model in project root
        default_model = os.path.abspath(os.path.join(os.path.dirname(__file__), "axsy-yolo.pt"))
        if not os.path.isfile(default_model):
            raise HTTPException(status_code=400, detail="Missing required header: detector (and default axsy-yolo.pt not found)")
        detector = default_model

    # Resolve model path locally or via GCS
    model_path = resolve_model_path(
        detector,
        gcs_bucket,
        customer_id=customer_id,
        model_id=model_id,
        project_id=project_id,
    )

    # Load model (cached) in threadpool to avoid blocking event loop
    model = await run_in_threadpool(load_model_cached, model_path)

    # Read file into PIL image (support both 'file' and 'image' fields)
    selected = file if file is not None else image
    if selected is None:
        raise HTTPException(status_code=400, detail="Missing image file. Use form field 'file' or 'image'.")
    try:
        contents = await selected.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    # Run inference in a thread and guard with per-model lock for safety
    inference = await run_in_threadpool(_run_inference_threadsafe, model_path, model, pil_image)

    # Optional product reclassification
    headers_flag = headers.get("reclassify_products") or headers.get("reclassify-products")
    flag = reclassify_products
    if flag is None and headers_flag is not None:
        val = str(headers_flag).strip().lower()
        flag = val in ("1", "true", "yes", "on")
    flag = bool(flag)
    if flag:
        try:
            # Resolve classifier path from header if provided; fallback to default local file
            clf_value = classifier
            clf_default = os.path.abspath(os.path.join(os.path.dirname(__file__), "axsy-classifier.pt"))
            if not clf_value:
                if not os.path.isfile(clf_default):
                    raise RuntimeError("Classifier weights not found.")
                classifier_path = clf_default
            else:
                classifier_path = resolve_model_path(
                    clf_value,
                    gcs_bucket,
                    customer_id=customer_id,
                    model_id=model_id,
                    project_id=project_id,
                )
            classifier_model = await run_in_threadpool(load_classifier_cached, classifier_path)
            labels = load_labels_for_classifier(
                clf_value if clf_value else classifier_path,
                classifier_path,
                gcs_bucket,
                customer_id,
                model_id,
                project_id,
                metadata_value=metadata,
            )
            re_stats = await run_in_threadpool(
                _reclassify_products,
                pil_image,
                inference.get("detections", []),
                classifier_path,
                classifier_model,
                labels,
            )
            if isinstance(re_stats, dict):
                inference["reclassify"] = re_stats
        except Exception as exc:
            logger.exception("Reclassify: skipped due to error: %s", exc)

    response: Dict[str, Any] = {
        "customer_id": customer_id,
        "model_id": model_id,
        "project_id": project_id,
        "detector": detector,
        "result": inference,
    }

    return JSONResponse(content=response)


@app.post("/classify")
async def classify(
    request: Request,
    file: Optional[UploadFile] = File(default=None, description="Image file to classify"),
    image: Optional[UploadFile] = File(default=None, description="Alternative field name for the image"),
    customer_id: Optional[str] = Header(default=None),
    model_id: Optional[str] = Header(default=None),
    project_id: Optional[str] = Header(default=None),
    classifier: Optional[str] = Header(default=None, description="Relative or absolute classifier path"),
    gcs_bucket: Optional[str] = Header(default=None, description="GCS bucket containing the classifier when path is a blob"),
    metadata: Optional[str] = Header(default=None, description="Optional path to metadata JSON containing labels"),
):
    headers = request.headers
    customer_id = customer_id or headers.get("customer_id") or headers.get("customer-id")
    model_id = model_id or headers.get("model_id") or headers.get("model-id")
    project_id = project_id or headers.get("project_id") or headers.get("project-id")
    classifier = classifier or headers.get("classifier") or headers.get("model")
    metadata = metadata or headers.get("metadata") or headers.get("meta")
    gcs_bucket = gcs_bucket or headers.get("gcs_bucket") or headers.get("gcs-bucket") or os.getenv("GCS_BUCKET")

    if classifier is None:
        default_model = os.path.abspath(os.path.join(os.path.dirname(__file__), "axsy-classifier.pt"))
        if not os.path.isfile(default_model):
            raise HTTPException(status_code=400, detail="Missing required header: classifier (and default axsy-classifier.pt not found)")
        classifier = default_model

    model_path = resolve_model_path(
        classifier,
        gcs_bucket,
        customer_id=customer_id,
        model_id=model_id,
        project_id=project_id,
    )

    model = await run_in_threadpool(load_classifier_cached, model_path)

    selected = file if file is not None else image
    if selected is None:
        raise HTTPException(status_code=400, detail="Missing image file. Use form field 'file' or 'image'.")
    try:
        contents = await selected.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}")

    inference = await run_in_threadpool(_run_classification_threadsafe, model_path, model, pil_image)
    from time import perf_counter
    t_post0 = perf_counter()
    labels = load_labels_for_classifier(
        classifier,
        model_path,
        gcs_bucket,
        customer_id,
        model_id,
        project_id,
        metadata_value=metadata,
    )
    top_label = None
    if labels:
        try:
            idx = int(inference.get("top_index", -1))
            if 0 <= idx < len(labels):
                top_label = labels[idx]
            else:
                top_label = labels[0]
        except Exception:
            top_label = labels[0]

    t_post1 = perf_counter()
    try:
        sp = inference.get("speed_ms") if isinstance(inference, dict) else None
        if isinstance(sp, dict):
            sp["postprocess"] = float((t_post1 - t_post0) * 1000.0)
    except Exception:
        pass

    response: Dict[str, Any] = {
        "customer_id": customer_id,
        "model_id": model_id,
        "project_id": project_id,
        "classifier": classifier,
        "result": {**inference, **({"top_label": top_label} if top_label is not None else {})},
    }

    return JSONResponse(content=response)


@app.post("/classify_batch")
async def classify_batch(
    request: Request,
    files: Optional[list[UploadFile]] = File(default=None, description="List of image files to classify"),
    customer_id: Optional[str] = Header(default=None),
    model_id: Optional[str] = Header(default=None),
    project_id: Optional[str] = Header(default=None),
    classifier: Optional[str] = Header(default=None, description="Relative or absolute classifier path"),
    gcs_bucket: Optional[str] = Header(default=None, description="GCS bucket containing the classifier when path is a blob"),
    metadata: Optional[str] = Header(default=None, description="Optional path to metadata JSON containing labels"),
    batch_size: Optional[int] = Query(default=16, ge=1, le=256, description="Max batch size for classification"),
):
    headers = request.headers
    customer_id = customer_id or headers.get("customer_id") or headers.get("customer-id")
    model_id = model_id or headers.get("model_id") or headers.get("model-id")
    project_id = project_id or headers.get("project_id") or headers.get("project-id")
    classifier = classifier or headers.get("classifier") or headers.get("model")
    metadata = metadata or headers.get("metadata") or headers.get("meta")
    gcs_bucket = gcs_bucket or headers.get("gcs_bucket") or headers.get("gcs-bucket") or os.getenv("GCS_BUCKET")

    if classifier is None:
        default_model = os.path.abspath(os.path.join(os.path.dirname(__file__), "axsy-classifier.pt"))
        if not os.path.isfile(default_model):
            raise HTTPException(status_code=400, detail="Missing required header: classifier (and default axsy-classifier.pt not found)")
        classifier = default_model

    model_path = resolve_model_path(
        classifier,
        gcs_bucket,
        customer_id=customer_id,
        model_id=model_id,
        project_id=project_id,
    )

    model = await run_in_threadpool(load_classifier_cached, model_path)

    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="Missing image files. Use form field 'files'.")

    # Resolve device and input size
    try:
        inp_size = int(model.sizes[0].item())
    except Exception:
        inp_size = 32
    try:
        device = next(model.parameters()).device.type
    except Exception:
        device = "cpu"

    # Read and decode images
    pil_images: list[Image.Image] = []
    indices: list[int] = []
    for idx, uf in enumerate(files):
        try:
            contents = await uf.read()
            img = Image.open(io.BytesIO(contents)).convert("RGB")
            pil_images.append(img)
            indices.append(idx)
        except Exception:
            pil_images.append(Image.new("RGB", (inp_size, inp_size)))
            indices.append(idx)

    # Batch through the list with the provided batch_size
    from time import perf_counter
    all_top_indices: list[int] = []
    all_top_probs: list[float] = []
    all_probs: list[list[float]] = []
    total_pre_ms = 0.0
    total_inf_ms = 0.0
    for start in range(0, len(pil_images), int(batch_size or 16)):
        chunk = pil_images[start : start + int(batch_size or 16)]
        t0 = perf_counter()
        batch_x = _prepare_image_tensor_batch(chunk, inp_size, device)
        t1 = perf_counter()
        from classifier import classify_image_batch
        out = classify_image_batch(model, batch_x)
        t2 = perf_counter()
        total_pre_ms += float((t1 - t0) * 1000.0)
        total_inf_ms += float((t2 - t1) * 1000.0)
        all_top_indices.extend([int(v) for v in out.get("top_indices", [])])
        all_top_probs.extend([float(v) for v in out.get("top_probs", [])])
        probs_list = out.get("probs", []) or []
        # Ensure nested lists of floats
        for p in probs_list:
            if isinstance(p, list):
                all_probs.append([float(x) for x in p])
            else:
                all_probs.append([])

    t_post0 = perf_counter()
    labels = load_labels_for_classifier(
        classifier,
        model_path,
        gcs_bucket,
        customer_id,
        model_id,
        project_id,
        metadata_value=metadata,
    )

    # Build per-item results
    results = []
    for i in range(len(pil_images)):
        idx_val = int(all_top_indices[i]) if i < len(all_top_indices) else -1
        prob_val = float(all_top_probs[i]) if i < len(all_top_probs) else 0.0
        top_label = None
        if labels and 0 <= idx_val < len(labels):
            top_label = labels[idx_val]
        elif labels:
            top_label = labels[0]
        results.append(
            {
                "top_index": idx_val,
                "top_prob": prob_val,
                **({"top_label": top_label} if top_label is not None else {}),
                **({"probs": all_probs[i]} if i < len(all_probs) else {}),
            }
        )
    t_post1 = perf_counter()
    total_post_ms = float((t_post1 - t_post0) * 1000.0)

    response: Dict[str, Any] = {
        "customer_id": customer_id,
        "model_id": model_id,
        "project_id": project_id,
        "classifier": classifier,
        "speed_ms": {"preprocess": total_pre_ms, "inference": total_inf_ms, "postprocess": total_post_ms},
        "results": results,
    }

    return JSONResponse(content=response)

@app.get("/", response_class=HTMLResponse)
async def upload_page():
    html = """
    <!doctype html>
    <html lang=\"en\">
    <head>
      <meta charset=\"utf-8\" />
      <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
      <title>Axsy Inference Demo</title>
      <style>
        :root { color-scheme: light dark; }
        body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica, Arial, \"Apple Color Emoji\", \"Segoe UI Emoji\"; margin: 0; padding: 24px; }
        h1 { margin: 0 0 12px; font-size: 20px; }
        .card { max-width: 1100px; margin: 0 auto; display: grid; grid-template-columns: 360px 1fr; gap: 24px; align-items: start; }
        .panel { border: 1px solid #9993; border-radius: 12px; padding: 16px; background: #fff2; backdrop-filter: blur(3px); }
        label { display: block; font-size: 12px; opacity: 0.8; margin: 10px 0 6px; }
        input[type=\"text\"], input[type=\"file\"], select { width: 100%; padding: 8px; border-radius: 8px; border: 1px solid #9996; background: transparent; }
        button { padding: 10px 14px; border-radius: 10px; border: 1px solid #6666; cursor: pointer; background: #0a7; color: white; font-weight: 600; }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
        .media { position: relative; max-width: 100%; }
        .media img { max-width: 100%; height: auto; display: block; border-radius: 10px; }
        canvas { position: absolute; left: 0; top: 0; pointer-events: none; }
        .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
        .details { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\", monospace; font-size: 12px; white-space: pre; background: #0000000b; border-radius: 10px; padding: 12px; overflow: auto; }
        .kv { display: grid; grid-template-columns: 180px 1fr; gap: 8px; font-size: 13px; }
        .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; border: 1px solid #9996; margin-right: 6px; }
      </style>
    </head>
    <body>
      <h1>Axsy Inference Demo</h1>
      <div class=\"card\">
        <div class=\"panel\">
          <form id=\"form\">
            <label>Image</label>
            <input id=\"file\" name=\"file\" type=\"file\" accept=\"image/*\" required />

            <label>Detector (model path or gs:// or blob path)</label>
            <input id=\"detector\" type=\"text\" placeholder=\"e.g. axsy-yolo.pt or gs://bucket/model.pt\" />

            <label>Classifier (weights path or gs:// or blob path)</label>
            <input id=\"classifier\" type=\"text\" placeholder=\"e.g. axsy-classifier.pt or gs://bucket/cls.pt\" />

            <label>Classifier metadata JSON (optional labels)</label>
            <input id=\"metadata\" type=\"text\" placeholder=\"e.g. axsy-classifier.json or gs://.../labels.json\" />

            <div class=\"row\">
              <div>
                <label>GCS bucket header</label>
                <input id=\"gcs_bucket\" type=\"text\" placeholder=\"bucket-name (optional)\" />
              </div>
              <div>
                <label>Customer ID</label>
                <input id=\"customer_id\" type=\"text\" placeholder=\"optional\" />
              </div>
            </div>
            <div class=\"row\">
              <div>
                <label>Model ID</label>
                <input id=\"model_id\" type=\"text\" placeholder=\"optional\" />
              </div>
              <div>
                <label>Project ID</label>
                <input id=\"project_id\" type=\"text\" placeholder=\"optional\" />
              </div>
            </div>

            <div style=\"display:flex; gap: 12px; align-items:center; margin-top:12px; flex-wrap: wrap;\">
              <button id=\"run\" type=\"submit\">Run inference</button>
              <label style=\"display:flex; gap:6px; align-items:center; margin:0;\"><input id=\"show_labels\" type=\"checkbox\" checked /> Show labels</label>
              <label style=\"display:flex; gap:6px; align-items:center; margin:0;\"><input id=\"reclassify_products\" type=\"checkbox\" /> Reclassify products</label>
              <label style=\"display:flex; gap:8px; align-items:center; margin:0;\">
                Overlay scale
                <input id=\"overlay_scale\" type=\"range\" min=\"0.5\" max=\"1.5\" step=\"0.05\" value=\"0.8\" />
                <span id=\"overlay_scale_value\">0.8x</span>
              </label>
            </div>
          </form>
        </div>

        <div class=\"panel\">
          <div class=\"media\" id=\"media\" style=\"display:none;\">
            <img id=\"img\" alt=\"uploaded image\" />
            <canvas id=\"overlay\"></canvas>
          </div>
          <div id=\"meta\" style=\"margin-top:12px; display:none;\"></div>
          <div id=\"details\" class=\"details\" style=\"display:none;\"></div>
        </div>
      </div>

      <script>
        const form = document.getElementById('form');
        const inputFile = document.getElementById('file');
        const runBtn = document.getElementById('run');
        const imgEl = document.getElementById('img');
        const canvas = document.getElementById('overlay');
        const media = document.getElementById('media');
        const meta = document.getElementById('meta');
        const details = document.getElementById('details');
        const showLabels = document.getElementById('show_labels');
        const overlayScale = document.getElementById('overlay_scale');
        const overlayScaleValue = document.getElementById('overlay_scale_value');
        const reclassifyProducts = document.getElementById('reclassify_products');

        function hashString(str) {
          let h = 0;
          for (let i = 0; i < str.length; i++) h = ((h << 5) - h + str.charCodeAt(i)) | 0;
          return h >>> 0;
        }
        function getClassHueMap(result) {
          const names = new Set(Object.keys(result.class_counts || {}));
          if ((!names || names.size === 0) && Array.isArray(result.detections)) {
            for (const det of result.detections) names.add(det.class_name);
          }
          const sorted = Array.from(names).sort();
          const n = Math.max(1, sorted.length);
          const map = {};
          for (let i = 0; i < sorted.length; i++) {
            // Evenly spaced hues with some offset for nicer defaults
            const hue = (i * 360 / n + 10) % 360;
            map[sorted[i]] = hue;
          }
          return map;
        }
        function colorForClass(name, alpha, hueMap) {
          const hue = hueMap && name in hueMap ? hueMap[name] : (hashString(String(name)) % 360);
          return `hsla(${hue}, 80%, 50%, ${alpha})`;
        }

        function clearOutput() {
          media.style.display = 'none';
          meta.style.display = 'none';
          details.style.display = 'none';
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

        function fmt(n) { return typeof n === 'number' ? n.toFixed(1) : n; }

        function drawDetections(imageDisplayWidth, imageDisplayHeight, response) {
          canvas.width = imageDisplayWidth;
          canvas.height = imageDisplayHeight;
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);

          const iw = response.result.image.width || imageDisplayWidth;
          const ih = response.result.image.height || imageDisplayHeight;
          const scaleX = imageDisplayWidth / iw;
          const scaleY = imageDisplayHeight / ih;

          const scaleFactor = overlayScale ? parseFloat(overlayScale.value) : 0.8;
          const baseLine = Math.max(1, Math.min(imageDisplayWidth, imageDisplayHeight) * 0.0012);
          const lineWidth = Math.max(1, baseLine * scaleFactor);
          ctx.lineWidth = lineWidth;
          const baseFont = Math.max(9, Math.round(imageDisplayWidth * 0.012));
          const fontPx = Math.max(9, Math.round(baseFont * scaleFactor));
          ctx.font = `${fontPx}px ui-monospace, monospace`;
          ctx.textBaseline = 'top';

          const hueMap = getClassHueMap(response.result);
          for (const det of response.result.detections) {
            const [x1, y1, x2, y2] = det.box.xyxy;
            const rx1 = x1 * scaleX, ry1 = y1 * scaleY, rx2 = x2 * scaleX, ry2 = y2 * scaleY;
            const w = rx2 - rx1, h = ry2 - ry1;
            const strokeColor = colorForClass(det.class_name, 1.0, hueMap);
            ctx.strokeStyle = strokeColor;
            ctx.strokeRect(rx1, ry1, w, h);

            if (showLabels.checked) {
              const label = `${det.class_name} ${fmt(det.confidence * 100)}%`;
              const padX = 6, padY = 3;
              const tw = ctx.measureText(label).width + padX * 2;
              const th = fontPx + padY * 2;
              const bx = Math.max(0, Math.min(rx1, imageDisplayWidth - tw));
              const by = Math.max(0, ry1 - th - 2);
              ctx.fillStyle = colorForClass(det.class_name, 0.85, hueMap);
              ctx.fillRect(bx, by, tw, th);
              ctx.fillStyle = '#fff';
              ctx.fillText(label, bx + padX, by + padY);
            }
          }
        }

        function renderMeta(response) {
          const r = response.result;
          const speed = r.speed_ms || {};
          const rc = r.reclassify || null;
          const rcSpeed = rc && rc.speed_ms ? rc.speed_ms : null;
          const pills = Object.entries(r.class_counts || {}).map(([k,v]) => `<span class=\"pill\">${k}: ${v}</span>`).join(' ');
          meta.innerHTML = `
            <div class=\"kv\">
              <div>Detections</div><div><strong>${r.num_detections}</strong></div>
              <div>Image</div><div>${r.image.width} × ${r.image.height}</div>
              <div>Speed (ms)</div><div>pre: ${fmt(speed.preprocess)} · infer: ${fmt(speed.inference)} · post: ${fmt(speed.postprocess)}</div>
              ${rc ? `<div>Reclassify</div><div>${rc.num} items</div>` : ''}
              ${rcSpeed ? `<div>Reclassify speed (ms)</div><div>pre: ${fmt(rcSpeed.preprocess)} · infer: ${fmt(rcSpeed.inference)} · post: ${fmt(rcSpeed.postprocess)}</div>` : ''}
              <div>Classes</div><div>${pills || '—'}</div>
            </div>
          `;
          meta.style.display = 'block';
        }

        function renderDetails(response) {
          details.textContent = JSON.stringify(response, null, 2);
          details.style.display = 'block';
        }

        async function runInference(evt) {
          evt.preventDefault();
          clearOutput();
          const file = inputFile.files && inputFile.files[0];
          if (!file) return;
          runBtn.disabled = true;
          try {
            const formData = new FormData();
            formData.append('file', file);

            const headers = {};
            const detector = document.getElementById('detector').value.trim();
            const classifier = document.getElementById('classifier').value.trim();
            const metadata = document.getElementById('metadata').value.trim();
            const gcs_bucket = document.getElementById('gcs_bucket').value.trim();
            const customer_id = document.getElementById('customer_id').value.trim();
            const model_id = document.getElementById('model_id').value.trim();
            const project_id = document.getElementById('project_id').value.trim();
            if (detector) headers['detector'] = detector;
            if (classifier) headers['classifier'] = classifier;
            if (metadata) headers['metadata'] = metadata;
            if (gcs_bucket) headers['gcs_bucket'] = gcs_bucket;
            if (customer_id) headers['customer_id'] = customer_id;
            if (model_id) headers['model_id'] = model_id;
            if (project_id) headers['project_id'] = project_id;
            if (reclassifyProducts && reclassifyProducts.checked) headers['reclassify_products'] = 'true';

            const resp = await fetch('/infer', { method: 'POST', body: formData, headers });
            if (!resp.ok) {
              const text = await resp.text();
              throw new Error(`HTTP ${resp.status}: ${text}`);
            }
            const data = await resp.json();

            // Show image
            imgEl.src = URL.createObjectURL(file);
            await imgEl.decode().catch(() => {});
            media.style.display = 'block';
            // Force canvas to follow displayed size
            canvas.style.width = imgEl.clientWidth + 'px';
            canvas.style.height = imgEl.clientHeight + 'px';
            drawDetections(imgEl.clientWidth, imgEl.clientHeight, data);
            renderMeta(data);
            renderDetails(data);
          } catch (err) {
            details.textContent = String(err);
            details.style.display = 'block';
          } finally {
            runBtn.disabled = false;
          }
        }

        form.addEventListener('submit', runInference);
        showLabels.addEventListener('change', () => {
          if (media.style.display !== 'none') {
            try { details.textContent && JSON.parse(details.textContent); } catch { return; }
            // If we have JSON, redraw using it
            try {
              const data = JSON.parse(details.textContent);
              drawDetections(imgEl.clientWidth, imgEl.clientHeight, data);
            } catch {}
          }
        });
        if (overlayScale && overlayScaleValue) {
          overlayScaleValue.textContent = overlayScale.value + 'x';
          overlayScale.addEventListener('input', () => {
            overlayScaleValue.textContent = overlayScale.value + 'x';
            if (media.style.display !== 'none') {
              try {
                const data = JSON.parse(details.textContent);
                drawDetections(imgEl.clientWidth, imgEl.clientHeight, data);
              } catch {}
            }
          });
        }
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


def get_app():
    return app


