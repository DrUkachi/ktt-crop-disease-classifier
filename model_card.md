<!--
Hugging Face model card — paste this file's contents as the README.md in the
Hub model repo at https://huggingface.co/DrUkachi/ktt-crop-disease-classifier.
The YAML front-matter is what the Hub uses to populate tags, task, and the
automatic "evaluation results" block.
-->
---
language:
  - en
license: mit
library_name: onnx
pipeline_tag: image-classification
tags:
  - agriculture
  - crop-disease
  - edge-ai
  - quantization
  - int8
  - mobilenetv3
  - onnx
datasets:
  - BrandonFors/Plant-Diseases-PlantVillage-Dataset
  - dpdl-benchmark/cassava
  - AI-Lab-Makerere/beans
metrics:
  - f1
model-index:
  - name: ktt-crop-disease-classifier
    results:
      - task:
          type: image-classification
          name: Crop disease classification (5-class)
        dataset:
          name: T2.1 synthetic recipe (PlantVillage + Cassava + iBeans, 300/class)
          type: custom
        metrics:
          - type: f1
            name: macro-F1 (clean test)
            value: 1.0000
          - type: f1
            name: macro-F1 (field-noisy test)
            value: 0.9867
---

# T2.1 — Compressed Crop Disease Classifier (INT8 ONNX, 4.34 MB)

A MobileNetV3-Small classifier compressed to 4.34 MB INT8 ONNX, trained for the
AIMS KTT Hackathon Tier 2 brief (T2.1). Takes a 224×224 JPEG leaf image, returns
one of five labels:

- `bean_spot` — bean angular leaf spot
- `cassava_mosaic` — Cassava Mosaic Disease (CMD)
- `healthy` — healthy maize leaf
- `maize_blight` — maize Northern Leaf Blight
- `maize_rust` — maize common rust

Intended for low-bandwidth, edge-device deployment in rural agricultural
contexts; shipped with a FastAPI/ONNX Runtime service and a USSD/SMS fallback
pathway for farmers on feature phones. See the GitHub repo for the full service,
Dockerfile, and product-adaptation artefact.

**GitHub:** [`DrUkachi/ktt-crop-disease-classifier`](https://github.com/DrUkachi/ktt-crop-disease-classifier)

---

## Evaluation

| Split | Macro-F1 | Notes |
|---|---|---|
| Clean test (150 imgs) | **1.0000** | balanced 30 per class |
| Field-noisy test (150 imgs) | **0.9867** | same images, blur σ ∈ [0, 1.5] + JPEG q ∈ [50, 85] + brightness jitter |
| Δ clean → field | **1.33 pp** | brief budget: < 12 pp ✅ |
| INT8 vs FP32 delta | **0.00 pp** | MatMul/Gemm-only INT8 is lossless on this backbone |

Per-class confusion matrices and Grad-CAM overlays are in [`notebooks/01_train_eval.ipynb`](https://github.com/DrUkachi/ktt-crop-disease-classifier/blob/master/notebooks/01_train_eval.ipynb).

**Honest caveat on clean F1 = 1.00.** PlantVillage (the source for the three
maize classes) is a studio-lit dataset with consistent per-class backgrounds,
and the 5 labels span 3 plant species with very different leaf morphology.
ImageNet-pretrained features separate those distributions trivially. The
**meaningful** number is the 1.33 pp drop on the field-noisy set — that measures
generalisation under blur, JPEG re-compression, and brightness jitter.

---

## Model details

- **Architecture:** MobileNetV3-Small, ImageNet pretrained, classifier head replaced with a `Linear(576 → 1024 → 5)` stack (the head is what 5-class fine-tunes).
- **Input:** 224 × 224 × 3 RGB, ImageNet mean/std normalization.
- **Output:** 5 logits in this fixed class ordering: `bean_spot, cassava_mosaic, healthy, maize_blight, maize_rust`.
- **Quantization:** ONNX Runtime dynamic INT8 on MatMul/Gemm nodes only (the classifier head), preceded by `quant_pre_process` (BN fusion, shape inference). Conv backbone stays FP32.
- **Why not full-graph INT8:** MobileNetV3's Hardswish activations + Squeeze-and-Excitation blocks regress catastrophically under ORT static INT8 (clean F1 → 0.73) and collapse entirely under full-graph dynamic INT8 (clean F1 → 0.07, always-one-class). QAT would fix this but was out of scope for the 4-hour brief cap. Full empirical table in the repo's [`process_log.md`](https://github.com/DrUkachi/ktt-crop-disease-classifier/blob/master/process_log.md) Hour 3.
- **Inference:** CPU-only via ONNX Runtime (`CPUExecutionProvider`). Observed latency ~3–5 ms per image.

## Training

- **Hardware:** NVIDIA L4 (23 GB). Full 15-epoch run took **40.2 seconds**.
- **Optimiser:** AdamW, LR 5e-4, weight decay 1e-4, cosine annealing over 15 epochs.
- **Loss:** class-weighted cross-entropy (weights all ≈ 1.0 for the balanced set, the scaffolding accommodates unbalanced splits).
- **Batch size:** 64.
- **Augmentation at train time:** horizontal flip, ±10° rotation, mild colour jitter (brightness/contrast 0.2, saturation 0.1). Blur and JPEG re-compression deliberately excluded so the clean→field gap reported is honest (those are the recipes applied to the field-noisy test set).
- **Best epoch:** 2 (val macro-F1 saturates early, training continues for finer convergence).

## Training data

Assembled by [`generate_dataset.py`](https://github.com/DrUkachi/ktt-crop-disease-classifier/blob/master/generate_dataset.py) from three public Hugging Face dataset mirrors:

| Class | HF dataset | Label |
|---|---|---|
| `bean_spot` | [`AI-Lab-Makerere/beans`](https://huggingface.co/datasets/AI-Lab-Makerere/beans) | idx 0 `angular_leaf_spot` |
| `cassava_mosaic` | [`dpdl-benchmark/cassava`](https://huggingface.co/datasets/dpdl-benchmark/cassava) | 3 CMD |
| `healthy` | [`BrandonFors/Plant-Diseases-PlantVillage-Dataset`](https://huggingface.co/datasets/BrandonFors/Plant-Diseases-PlantVillage-Dataset) | idx 10 `Corn_(maize)___healthy` |
| `maize_blight` | same | idx 9 `Corn_(maize)___Northern_Leaf_Blight` |
| `maize_rust` | same | idx 8 `Corn_(maize)___Common_rust_` |

300 images per class, 80/10/10 train/val/test split (seed 1337). Full
provenance (per-image source IDs) is recorded in `data/manifest.json` after the
generator runs.

## Usage

### With ONNX Runtime directly

```python
import numpy as np, onnxruntime as ort
from PIL import Image

CLASSES = ["bean_spot", "cassava_mosaic", "healthy", "maize_blight", "maize_rust"]
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

sess = ort.InferenceSession("model.onnx", providers=["CPUExecutionProvider"])
img = Image.open("maize_rust.jpg").convert("RGB").resize((224, 224))
arr = (np.asarray(img, dtype=np.float32) / 255.0 - MEAN) / STD
arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)
logits = sess.run(None, {sess.get_inputs()[0].name: arr})[0][0]
print(CLASSES[int(logits.argmax())])
```

### As a FastAPI service (from the GitHub repo)

```bash
git clone https://github.com/DrUkachi/ktt-crop-disease-classifier.git
cd ktt-crop-disease-classifier
pip install -r service/requirements.txt
uvicorn service.app:app --host 0.0.0.0 --port 8000
# then:
curl -X POST -F 'image=@samples/maize_rust_1.jpg' http://localhost:8000/predict
```

The service returns `{ label, confidence, top3, latency_ms, rationale }` and
adds `escalation: "second_photo_different_angle"` when `confidence < 0.6`.

---

## Limitations and intended use

- Trained on ~1,200 studio-lit and smartphone-quality images. Performance on
  microscope, UV, or non-leaf substrate images is not characterised.
- The five classes do not cover all realistic field scenarios — a farmer with a
  *tomato* leaf or a *coffee berry disease* will get a confident wrong answer.
  The service exposes `top3` and an `escalation` field so the consuming PWA
  (see [`ussd_fallback.md`](https://github.com/DrUkachi/ktt-crop-disease-classifier/blob/master/ussd_fallback.md)) can route low-confidence cases to a human extension officer.
- Training data provenance is inherited from the upstream HF mirrors. If a
  mirror removes an image, re-running the generator will produce a slightly
  different sample. The `manifest.json` records per-image source IDs for
  reproducibility.
- The model card does *not* evaluate fairness across cultivars, soil types, or
  geographies — the source datasets are not annotated with that metadata.

## License

[MIT](https://github.com/DrUkachi/ktt-crop-disease-classifier/blob/master/LICENSE), matching the GitHub repo.

## Citation

```bibtex
@misc{osisiogu2026ktt,
  author = {Osisiogu, Ukachi},
  title  = {Compressed Crop Disease Classifier (AIMS KTT T2.1)},
  year   = {2026},
  howpublished = {\url{https://github.com/DrUkachi/ktt-crop-disease-classifier}},
}
```

Upstream dataset credits: PlantVillage (Mohanty et al. 2016), Cassava Leaf
Disease (Mwebaze et al. 2019, Kaggle 2020), iBeans (Makerere AI Lab 2020).
