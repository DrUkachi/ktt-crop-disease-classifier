"""FastAPI inference service for the compressed crop-disease classifier.

POST /predict
    multipart form: image=<JPEG bytes>
    response JSON: { label, confidence, top3, latency_ms, rationale[, escalation] }

Two modes, picked automatically at startup:

* **Full mode** — torch + torchvision + `checkpoints/best.pt` are present.
  Rationale carries a Grad-CAM-derived attention summary ("attention
  upper-left, covers 18% of leaf") plus the class cue. Used for the live demo
  and the evaluation notebook.
* **Lightweight mode** — ONNX Runtime only. What ships in the Docker image.
  Rationale falls back to (class cue + top-2 margin), no torch import.

When `confidence < 0.6` an `"escalation": "second_photo_different_angle"` field
is added — matches the low-confidence path in `ussd_fallback.md`'s PWA design.
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "model.onnx"
CHECKPOINT_PATH = REPO_ROOT / "checkpoints" / "best.pt"
CLASSES = ["bean_spot", "cassava_mosaic", "healthy", "maize_blight", "maize_rust"]
INPUT_SIZE = 224
ESCALATION_THRESHOLD = 0.6

# ImageNet normalization — backbone was pretrained on ImageNet.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_CLASS_CUE = {
    "maize_rust":     "lesion density consistent with rust pustules",
    "maize_blight":   "elongated greyish lesions along the leaf veins",
    "cassava_mosaic": "mottled chlorotic patches with leaf distortion",
    "bean_spot":      "angular brown spots bounded by leaf veins",
    "healthy":        "uniform green, no visible lesions or chlorosis",
}

app = FastAPI(title="Crop Disease Classifier", version="0.1.0")

_session: ort.InferenceSession | None = None


# ---------------------------------------------------------------------------
# Optional full-mode (PyTorch + Grad-CAM) — loaded lazily at first /predict.
# We resolve "is full mode available?" up-front so /health can report it, but
# the actual torch model load is deferred until someone hits /predict — that
# keeps cold-start cheap when no one is asking for rationales.
# ---------------------------------------------------------------------------

def _full_mode_available() -> bool:
    if not CHECKPOINT_PATH.exists():
        return False
    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        return False
    return True


_FULL_MODE = _full_mode_available()
_torch_model = None  # loaded lazily on first call
_gradcam_fn = None


def _load_torch_model():
    """Lazy import + load — only if full mode is on and someone's asking."""
    global _torch_model, _gradcam_fn
    if _torch_model is not None:
        return _torch_model, _gradcam_fn

    import torch
    from torchvision.models import mobilenet_v3_small

    from service.gradcam import GradCAM, heatmap_summary

    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(CLASSES))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _torch_model = model
    _gradcam_fn = (GradCAM, heatmap_summary, torch)
    return _torch_model, _gradcam_fn


def _load_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"model.onnx not found at {MODEL_PATH}. "
                "Train and export first: python train.py && python export_onnx.py"
            )
        _session = ort.InferenceSession(
            str(MODEL_PATH), providers=["CPUExecutionProvider"]
        )
    return _session


def _preprocess(jpeg_bytes: bytes) -> np.ndarray:
    """JPEG bytes -> (1, 3, 224, 224) float32, ImageNet-normalised."""
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr.astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _rationale_lightweight(label: str, confidence: float, probs: np.ndarray) -> str:
    cue = _CLASS_CUE.get(label, "top class score dominates the distribution")
    margin = float(confidence - sorted(probs, reverse=True)[1])
    return f"{cue}; top-2 margin {margin:.2f}"


def _rationale_full(label: str, confidence: float, probs: np.ndarray, x_np: np.ndarray) -> str:
    """Grad-CAM-derived rationale. Falls back to lightweight if anything fails."""
    try:
        model, (GradCAM, heatmap_summary, torch) = _load_torch_model()
        x_tensor = torch.from_numpy(x_np)
        with GradCAM(model) as gc:
            res = gc.compute(x_tensor)
        summary = heatmap_summary(res.heatmap)
        cue = _CLASS_CUE.get(label, "top class score dominates the distribution")
        margin = float(confidence - sorted(probs, reverse=True)[1])
        area_pct = int(round(summary["area_gt50"] * 100))
        quadrant = summary["quadrant"]
        return (
            f"attention {quadrant} (covers {area_pct}% of leaf); "
            f"{cue}; top-2 margin {margin:.2f}"
        )
    except Exception:
        # If anything goes sideways in the rationale path we should not kill
        # the /predict call — the label and confidence still need to ship.
        return _rationale_lightweight(label, confidence, probs)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "rationale_mode": "full" if _FULL_MODE else "lightweight",
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict:
    if image.content_type not in {"image/jpeg", "image/jpg", "image/png"}:
        raise HTTPException(415, f"unsupported content_type: {image.content_type}")

    jpeg = await image.read()
    if not jpeg:
        raise HTTPException(400, "empty image payload")

    try:
        x = _preprocess(jpeg)
    except Exception as e:
        raise HTTPException(400, f"preprocessing failed: {e}") from e

    sess = _load_session()
    input_name = sess.get_inputs()[0].name

    t0 = time.perf_counter()
    logits = sess.run(None, {input_name: x})[0][0]
    latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    probs = _softmax(logits)
    order = np.argsort(probs)[::-1]
    top3 = [{"label": CLASSES[i], "score": float(probs[i])} for i in order[:3]]
    top1 = top3[0]
    label = top1["label"]
    confidence = top1["score"]

    if _FULL_MODE:
        rationale = _rationale_full(label, confidence, probs, x)
    else:
        rationale = _rationale_lightweight(label, confidence, probs)

    response: dict = {
        "label": label,
        "confidence": confidence,
        "top3": top3,
        "latency_ms": latency_ms,
        "rationale": rationale,
    }
    if confidence < ESCALATION_THRESHOLD:
        response["escalation"] = "second_photo_different_angle"
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
