# Compressed Crop Disease Classifier (T2.1)

> AIMS KTT Hackathon · Tier 2 · Edge-AI for Offline Crop Diagnostics
> 5 classes · MobileNetV3-Small backbone · INT8 ONNX (< 10 MB) · FastAPI service · USSD fallback

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

_(Substitute `bash run.sh` once the wrapper script is in.)_

---

## Repo layout

```
.
├── notebooks/                 # exploratory notebook(s) — training + eval
├── service/
│   ├── app.py                 # FastAPI /predict endpoint
│   ├── Dockerfile
│   └── requirements.txt
├── samples/                   # example JPEGs to curl against /predict
├── generate_dataset.py        # synthetic-recipe regenerator (PlantVillage + Cassava)
├── model.onnx                 # final INT8 ONNX (< 10 MB) — added after training
├── README.md
├── ussd_fallback.md           # Product & Business artifact
├── process_log.md             # hour-by-hour timeline + LLM declarations
├── SIGNED.md                  # honor-code acknowledgement
└── LICENSE                    # MIT
```

---

## Classes

| Label             | Source mirror                   | Approx images |
|-------------------|---------------------------------|---------------|
| `healthy`         | PlantVillage (maize healthy)    | ~300          |
| `maize_rust`      | PlantVillage (Maize_Common_rust)| ~300          |
| `maize_blight`    | PlantVillage (Maize_NLB)        | ~300          |
| `cassava_mosaic`  | Cassava Leaf Disease (CMD)      | ~300          |
| `bean_spot`       | PlantVillage (Bean_ALS)         | ~300          |

**Split:** 80 / 10 / 10 train / val / test.
**Field-noisy variant** (`test_field/`): same test images with random blur
(σ ∈ [0, 1.5]), JPEG compression (q ∈ [50, 85]), and brightness jitter.

---

## Model

- **Backbone:** MobileNetV3-Small (ImageNet pre-trained), fine-tuned head.
- **Input:** 224 × 224 RGB.
- **Quantization:** post-training static INT8 (ONNX Runtime).
- **Final size:** _TBD — target < 10 MB_.

### Reported metrics

| Split             | Macro-F1 |
|-------------------|----------|
| Clean test        | _TBD_    |
| Field-noisy test  | _TBD_    |
| Δ (drop)          | _< 12 pp target_ |

---

## Service

`service/app.py` exposes `POST /predict` (multipart `image=<JPEG>`):

```bash
curl -X POST -F 'image=@samples/maize_rust_1.jpg' http://localhost:8000/predict
```

```jsonc
{
  "label": "maize_rust",
  "confidence": 0.93,
  "top3": [
    {"label": "maize_rust",   "score": 0.93},
    {"label": "maize_blight", "score": 0.04},
    {"label": "healthy",      "score": 0.02}
  ],
  "latency_ms": 41,
  "rationale": "high lesion density consistent with rust pustules"
}
```

Run via Docker:

```bash
docker build -t crop-clf service/
docker run --rm -p 8000:8000 crop-clf
```

---

## Product & Business adaptation

See [ussd_fallback.md](ussd_fallback.md) for the 3-step relay workflow,
Kinyarwanda + French SMS templates, and 1,000-farmer unit economics.

---

## Submission artefacts

- **Repo:** _this repo_
- **Model checkpoint:** _Hugging Face Hub link — TBD_
- **4-minute video:** _YouTube unlisted — TBD_
- **Dataset generator:** [`generate_dataset.py`](generate_dataset.py)

---

## Honor code

I declared every LLM tool I used in [`process_log.md`](process_log.md) and
signed the honor code in [`SIGNED.md`](SIGNED.md).

License: [MIT](LICENSE).
