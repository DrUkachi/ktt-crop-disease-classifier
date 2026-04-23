"""FastAPI inference service for the compressed crop-disease classifier.

POST /predict
    multipart form: image=<JPEG bytes>
    response JSON: { label, confidence, top3, latency_ms, rationale }
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

MODEL_PATH = Path(__file__).resolve().parent.parent / "model.onnx"
CLASSES = ["bean_spot", "cassava_mosaic", "healthy", "maize_blight", "maize_rust"]
INPUT_SIZE = 224

# ImageNet normalization — backbone was pretrained on ImageNet.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

app = FastAPI(title="Crop Disease Classifier", version="0.1.0")

_session: ort.InferenceSession | None = None


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
    img = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - _MEAN) / _STD
    arr = arr.transpose(2, 0, 1)[None, ...]  # NCHW
    return arr.astype(np.float32)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _rationale(label: str, confidence: float, probs: np.ndarray) -> str:
    """Lightweight, deterministic rationale string.

    A future iteration will swap this for a Grad-CAM attribution summary
    (stretch goal in the brief). For now we ship a hand-written cue per class
    so the extension officer reading the JSON has something to relay.
    """
    cue = {
        "maize_rust": "lesion density consistent with rust pustules",
        "maize_blight": "elongated greyish lesions along the leaf veins",
        "cassava_mosaic": "mottled chlorotic patches with leaf distortion",
        "bean_spot": "angular brown spots bounded by leaf veins",
        "healthy": "uniform green, no visible lesions or chlorosis",
    }.get(label, "top class score dominates the distribution")
    margin = float(confidence - sorted(probs, reverse=True)[1])
    return f"{cue}; top-2 margin {margin:.2f}"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


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

    return {
        "label": top1["label"],
        "confidence": top1["score"],
        "top3": top3,
        "latency_ms": latency_ms,
        "rationale": _rationale(top1["label"], top1["score"], probs),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
