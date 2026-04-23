# Compressed Crop Disease Classifier (T2.1)

> AIMS KTT Hackathon · Tier 2 · Edge-AI for Offline Crop Diagnostics
> 5 classes · MobileNetV3-Small backbone · **INT8 ONNX, 4.34 MB** · macro-F1 **1.000 clean / 0.987 field-noisy** · FastAPI service · Grad-CAM rationale · USSD fallback

[**Model on Hugging Face Hub →** `DrUkachi/ktt-crop-disease-classifier`](https://huggingface.co/DrUkachi/ktt-crop-disease-classifier)

A compact image classifier that tells a farmer whether a maize, cassava, or
bean leaf is **healthy**, has **maize_rust**, **maize_blight**,
**cassava_mosaic**, or **bean_spot** — and a non-smartphone delivery path so
the diagnosis still lands when the user has only a feature phone.

---

## Reproduce in ≤ 2 commands (free Colab CPU)

```bash
pip install -r requirements.txt
python generate_dataset.py --out data/ && python train.py && python export_onnx.py
```

`train.py` auto-detects the device (`cuda` if a GPU is attached, else `cpu`) so
the same commands work on Colab CPU free-tier (~30 min end-to-end). Inference
and the `/predict` service are CPU-only via ONNX Runtime regardless.

---

## How to use

Three paths — pick the one that matches what you want to verify.

> **Full vs lightweight mode.** The FastAPI service picks its mode automatically
> at startup based on what's on disk — there is no flag to set.
>
> - **Full mode** (Grad-CAM rationale in the `/predict` JSON) requires both
>   `checkpoints/best.pt` *and* PyTorch installed. Path A (which trains the
>   model) produces `best.pt`; `pip install -r requirements.txt` installs torch.
>   `GET /health` returns `"rationale_mode": "full"`.
> - **Lightweight mode** (ONNX Runtime only, class-cue rationale) runs whenever
>   the checkpoint is missing OR PyTorch isn't installed. A fresh `git clone`
>   starts in lightweight mode — `checkpoints/` is gitignored. Paths B, C, and
>   the Docker image ship lightweight by default. `GET /health` returns
>   `"rationale_mode": "lightweight"`.
>
> The label / confidence / top3 / latency numbers are bit-identical across modes
> — only the `rationale` string differs. See the [Service](#service) section for
> the full JSON shape for each mode.

### A. Full reproduce on Google Colab (free CPU, ~30 min)

Exactly what the brief's evaluators will execute. Open a fresh Colab notebook,
set the runtime to **CPU** (`Runtime → Change runtime type → CPU`), then run
these four cells:

```python
# Cell 1 — clone
!git clone https://github.com/DrUkachi/ktt-crop-disease-classifier.git
%cd ktt-crop-disease-classifier
```

```python
# Cell 2 — install
!pip install -r requirements.txt
```

```python
# Cell 3 — the 2-command reproduce line, verbatim
!python generate_dataset.py --out data/ && python train.py && python export_onnx.py
```

```python
# Cell 4 — verify the three brief gates
!ls -la model.onnx                  # < 10 MB
!cat checkpoints/metrics.json       # macro-F1 >= 0.80, clean→field drop < 12 pp
```

Expected end state: `model.onnx` ≈ **4.34 MB**, macro-F1 **1.0000** clean /
**0.9867** field-noisy, Δ **1.33 pp**. Training on Colab free CPU takes ~30
minutes (vs 40 s on the L4 this was developed on).

### B. Inference-only demo (no training, ~3 min)

Uses the committed `model.onnx` and exercises the FastAPI `/predict` endpoint.
Works on Colab CPU, any laptop, or inside the Docker image.

```python
# Colab cells
!git clone https://github.com/DrUkachi/ktt-crop-disease-classifier.git
%cd ktt-crop-disease-classifier
!pip install -r service/requirements.txt
```

```python
import subprocess, time
proc = subprocess.Popen(
    ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
)
time.sleep(8)  # let uvicorn bind
```

```python
!curl -s http://localhost:8000/health
!curl -s -X POST -F 'image=@samples/maize_rust_1.jpg' http://localhost:8000/predict
```

Without `checkpoints/best.pt` present the service runs in **lightweight mode**
(ONNX Runtime only, class-cue rationale). JSON schema is identical — see the
[Service](#service) section for the full response shape.

### C. Python API — load `model.onnx` directly

No service, no Docker, just a numpy + ONNX Runtime call. Useful for scripting
or embedding into a notebook:

```python
import numpy as np, onnxruntime as ort
from PIL import Image

CLASSES = ["bean_spot", "cassava_mosaic", "healthy", "maize_blight", "maize_rust"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
img  = Image.open("samples/maize_rust_1.jpg").convert("RGB").resize((224, 224))
arr  = ((np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
logits = sess.run(None, {sess.get_inputs()[0].name: arr.astype(np.float32)})[0][0]
probs  = np.exp(logits) / np.exp(logits).sum()
print(CLASSES[int(probs.argmax())], float(probs.max()))
```

### Docker (lightweight mode)

```bash
docker build -t crop-clf service/
docker run --rm -p 8000:8000 crop-clf
curl -X POST -F 'image=@samples/maize_rust_1.jpg' http://localhost:8000/predict
```

---

## Repo layout

```
.
├── notebooks/
│   └── 01_train_eval.ipynb    # confusion matrices, training curves, Grad-CAM
├── service/
│   ├── app.py                 # FastAPI /predict — dual-mode (see below)
│   ├── gradcam.py             # Grad-CAM for the MobileNetV3-Small backbone
│   ├── Dockerfile             # ships the lightweight mode
│   └── requirements.txt
├── samples/                   # example JPEGs to curl against /predict
├── generate_dataset.py        # synthetic-recipe regenerator (HF mirrors)
├── train.py                   # MobileNetV3-Small fine-tune, device auto-detect
├── export_onnx.py             # FP32 ONNX -> INT8 on MatMul/Gemm, size-gated
├── model.onnx                 # final INT8 ONNX, 4.34 MB (submission artefact)
├── checkpoints/               # best.pt, fp32.onnx, metrics.json (gitignored)
├── README.md
├── ussd_fallback.md           # Product & Business artefact
├── process_log.md             # hour-by-hour timeline + LLM declarations
├── SIGNED.md                  # honor-code acknowledgement
└── LICENSE                    # MIT
```

---

## Classes

| Label             | HF source mirror                                                   | Label idx | Images |
|-------------------|---------------------------------------------------------------------|-----------|--------|
| `bean_spot`       | [`AI-Lab-Makerere/beans`](https://huggingface.co/datasets/AI-Lab-Makerere/beans) | 0 — `angular_leaf_spot` | 300 |
| `cassava_mosaic`  | [`dpdl-benchmark/cassava`](https://huggingface.co/datasets/dpdl-benchmark/cassava) | 3 — CMD | 300 |
| `healthy`         | [`BrandonFors/Plant-Diseases-PlantVillage-Dataset`](https://huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset) | 10 — `Corn_(maize)___healthy` | 300 |
| `maize_blight`    | same as above                                                       | 9 — `Corn_(maize)___Northern_Leaf_Blight` | 300 |
| `maize_rust`      | same as above                                                       | 8 — `Corn_(maize)___Common_rust_` | 300 |

**Split:** 80 / 10 / 10 train / val / test → **1,200 / 150 / 150** images.
**Field-noisy variant** (`data/test_field/`, 150 images): the same 150 test images
with random Gaussian blur (σ ∈ [0, 1.5]), JPEG re-compression (q ∈ [50, 85]),
and brightness jitter applied per the brief's recipe.

> The scaffold originally pointed at three HF mirrors that have since gone
> dead. The swap to the mirrors above is documented in
> [`process_log.md`](process_log.md) Hour 1.

---

## Model

- **Backbone:** MobileNetV3-Small (ImageNet pre-trained), end-to-end fine-tuned.
- **Input:** 224 × 224 RGB, ImageNet normalization.
- **Training:** AdamW + cosine LR, batch 64, 15 epochs, class-weighted CE. Light
  augmentation (hflip, ±10° rotation, mild colour jitter). Training run took
  **40.2 s** on an NVIDIA L4. Best val macro-F1 hit at **epoch 2**.
- **Quantization:** ONNX Runtime dynamic INT8 applied to **MatMul/Gemm nodes only**
  (classifier head), with `quant_pre_process` (BN fusion + shape inference) first.
  Conv backbone stays FP32 — MobileNetV3's Hardswish + Squeeze-Excitation blocks
  don't survive full-graph INT8 on ORT without QAT (see Hour 3 of
  [`process_log.md`](process_log.md) for the empirical comparison).
- **Final artefact:** [`model.onnx`](model.onnx) — **4.34 MB** INT8.

### Reported metrics

| Split                    | Macro-F1 (INT8 `model.onnx`) | Budget      | Status |
|--------------------------|------------------------------|-------------|--------|
| Clean test               | **1.0000**                   | ≥ 0.80      | ✅     |
| Field-noisy test         | **0.9867**                   | —           |        |
| Δ clean → field          | **1.33 pp**                  | < 12 pp     | ✅     |
| INT8 vs FP32 clean delta | **0.00 pp**                  | —           |        |
| File size                | **4.34 MB**                  | < 10 MB     | ✅     |

All three brief gates satisfied. `export_onnx.py` asserts these budgets at
export time — it will unlink `model.onnx` and fail the build if any regress.
Per-class breakdown and confusion matrices are in
[`notebooks/01_train_eval.ipynb`](notebooks/01_train_eval.ipynb) (rendered on GitHub).

### Honest reading of "F1 = 1.00"

PlantVillage is a studio-lit dataset with consistent backgrounds per class, and
our 5 labels span 3 plant species with very different leaf morphology (maize
narrow, cassava palmate, beans oval). ImageNet-pretrained features separate
those distributions trivially, so the clean-test F1 plateaus at 1.0 by
epoch 2. The **meaningful** number is the 1.33 pp drop on the field-noisy set
— that's the one that actually measures generalisation under blur, JPEG
re-compression, and brightness jitter. `data/manifest.json` records 300 unique
source IDs per class with zero overlap across train/val/test, so the perfect
clean score isn't leakage.

---

## Service

`service/app.py` exposes `POST /predict` (multipart `image=<JPEG>`). The service
runs in one of two modes picked automatically at startup; the JSON response
shape is identical, the `rationale` field differs.

### Full mode — `checkpoints/best.pt` + PyTorch available

Used for the live demo and the eval notebook. Runs ONNX Runtime for
label/confidence/top3, then **Grad-CAM** on the FP32 PyTorch checkpoint to
derive a model-attention summary. Adds ~80 ms per request but every response
carries real model evidence.

```bash
curl -X POST -F 'image=@samples/maize_rust_1.jpg' http://localhost:8000/predict
```

```jsonc
{
  "label": "maize_rust",
  "confidence": 1.00,
  "top3": [
    {"label": "maize_rust",   "score": 1.00},
    {"label": "maize_blight", "score": 0.00},
    {"label": "cassava_mosaic", "score": 0.00}
  ],
  "latency_ms": 4.4,
  "rationale": "attention centre (covers 34% of leaf); lesion density consistent with rust pustules; top-2 margin 1.00"
}
```

`latency_ms` measures the ORT `session.run` call only, not the Grad-CAM step
— the brief's latency number is the inference artefact's speed.

### Lightweight mode — ONNX Runtime only (Docker default)

What ships in the container. Drops PyTorch + the checkpoint, keeps the image
small (~200 MB vs ~2 GB in full mode). Rationale falls back to a class cue +
top-2 margin.

```jsonc
{
  "label": "maize_rust",
  "confidence": 1.00,
  "top3": [ /* same */ ],
  "latency_ms": 4.7,
  "rationale": "lesion density consistent with rust pustules; top-2 margin 1.00"
}
```

`GET /health` reports which mode is active:

```jsonc
{"status": "ok", "model_loaded": true, "rationale_mode": "full"}
```

### Low-confidence escalation

When `confidence < 0.6`, both modes add an
`"escalation": "second_photo_different_angle"` field. The feature-phone PWA
described in [`ussd_fallback.md`](ussd_fallback.md) uses this as the trigger
to prompt the village agent for a second capture.

For the Docker (lightweight mode) run-path, see [How to use →
Docker](#docker-lightweight-mode) above.

---

## Product & Business adaptation

See [ussd_fallback.md](ussd_fallback.md) for the 3-step relay workflow,
Kinyarwanda + French SMS templates, and 1,000-farmer unit economics.

---

## Submission artefacts

- **Repo:** this repo — [`DrUkachi/ktt-crop-disease-classifier`](https://github.com/DrUkachi/ktt-crop-disease-classifier).
- **Model:** [`model.onnx`](model.onnx) at the repo root — INT8 ONNX, 4.34 MB.
  Mirrored on Hugging Face Hub at [`DrUkachi/ktt-crop-disease-classifier`](https://huggingface.co/DrUkachi/ktt-crop-disease-classifier) with a full model card (YAML-indexed for task / license / metrics).
- **4-minute video:** _TBD — YouTube unlisted_ (see [`process_log.md`](process_log.md) for the segment script).
- **Eval notebook:** [`notebooks/01_train_eval.ipynb`](notebooks/01_train_eval.ipynb) — rendered inline on GitHub.
- **Dataset generator:** [`generate_dataset.py`](generate_dataset.py) — streams from 3 HF mirrors.
- **Product & business artefact:** [`ussd_fallback.md`](ussd_fallback.md) — 3-step feature-phone relay, Kinyarwanda+French SMS template, 1,000-farmer unit economics.
- **Process log:** [`process_log.md`](process_log.md) — hour-by-hour timeline, declared LLM prompts, hardest-decision paragraph.
- **Signed honor code:** [`SIGNED.md`](SIGNED.md).

---

## Honor code

I declared every LLM tool I used in [`process_log.md`](process_log.md) and
signed the honor code in [`SIGNED.md`](SIGNED.md).

License: [MIT](LICENSE).
