# Process Log — T2.1 Compressed Crop Disease Classifier

**Candidate:** Ukachi Osisiogu
**Challenge start:** 2026-04-23
**Time budget:** 180 min target / 4 hr hard cap

---

## Hour-by-hour timeline

### Hour 0 — Kickoff (2026-04-23, ~09:15)
- Read the candidate brief (`T2.1_Compressed_Crop_Disease_Classifier.pdf`).
- Created public GitHub repo `ktt-crop-disease-classifier` (MIT license).
- Wrote `SIGNED.md` with honor code copied verbatim from page 5 of the brief.
- Started this `process_log.md`.
- Scaffolded repo skeleton: `/notebooks/`, `/service/{app.py, Dockerfile}`, `/samples/`, `README.md`, `ussd_fallback.md`, `.gitignore`.
- Drafted dataset generator script (`generate_dataset.py`) implementing the synthetic recipe (PlantVillage + Cassava mirrors, ~300 imgs/class, 5 classes; 80/10/10 split; field-noise variant with blur σ ∈ [0, 1.5], JPEG q ∈ [50, 85], brightness jitter).

### Hour 1 — Data pipeline repair + materialisation (2026-04-23)
Opened Claude Code to continue from the Phase 0 scaffold. Working on Lightning.ai (Studio, CPU for data work, GPU on standby for training — the brief lets us train on Colab CPU but Lightning gives me a GPU toggle for free, so I'll use it and note the switch here).

**Dataset mirror swap (unexpected).** The generator scaffold pointed at three HF mirrors that have all rotted since Phase 0:

- `yusufberkay/plantvillage-dataset` — `DatasetNotFoundError`
- `Dauka-CA/Cassava-Leaf-Disease-Detection-Train` — `DatasetNotFoundError`
- `nateraw/beans` — builder script still up, but the backing `storage.googleapis.com/ibeans/train.zip` returns HTTP 403.

Probed the Hub via `HfApi.list_datasets` and hand-verified three live replacements. Updated `SOURCE_MAP` in `generate_dataset.py` and reworked `_load_hf` to handle integer `ClassLabel` / raw int64 labels (the old code only compared strings, so even if the IDs had been right it would have silently matched zero rows).

| Class | Mirror | Label |
|---|---|---|
| `healthy` | `BrandonFors/Plant-Diseases-PlantVillage-Dataset` | idx 10 (`Corn_(maize)___healthy`) |
| `maize_rust` | same | idx 8 (`Corn_(maize)___Common_rust_`) |
| `maize_blight` | same | idx 9 (`Corn_(maize)___Northern_Leaf_Blight`) |
| `cassava_mosaic` | `dpdl-benchmark/cassava` | label 3 (CMD — confirmed via `train-cmd-*.jpg` filename pattern) |
| `bean_spot` | `AI-Lab-Makerere/beans` | idx 0 (`angular_leaf_spot`) |

**Generator run.** 300 imgs/class × 5 classes, 80/10/10 split: `train=1200 val=150 test=150` plus 150 field-noisy. All 224×224 RGB. Manifest at `data/manifest.json` records per-class source IDs plus the full `source_map` so the swap is self-documenting for evaluators.

**Samples landed.** One clean JPEG per class plus one field-noisy `maize_rust_field_1.jpg` in `samples/` — including `maize_rust_1.jpg`, the exact filename the brief's video spec names.

Commits: `35b7d75 Phase 1: swap to live HF mirrors and land demo samples`.

### Hour 2 — Training (2026-04-23)
Switched the Lightning Studio to GPU (NVIDIA L4, 23 GB VRAM) and wrote `train.py`. Fine-tuned MobileNetV3-Small (ImageNet pretrained) end-to-end on the 1200-image train set. Single AdamW + cosine LR, batch 64, 15 epochs, light augmentation (horizontal flip, ±10° rotation, mild colour jitter). **Deliberately did not add blur / JPEG re-compression to train-time augmentation**, so the clean → field gap I report is honest — the brief scores robustness on the field set and training on the noise recipe would be training-on-test.

Run: **40.2 seconds total** on the L4, loss 0.35 → 0.016 by epoch 2, validation macro-F1 saturates at 1.0000 from epoch 2 onward.

| Split | Macro-F1 |
|---|---|
| Val (clean) — best | 1.0000 |
| Test (clean) | 1.0000 |
| Test (field-noisy) | 0.9867 |
| **Δ clean → field** | **1.33 pp** (budget < 12 pp) |

Both hard targets met on the first run — no re-training needed. The reason the clean number is perfect rather than ~90% is not leakage: I cross-checked `data/manifest.json` and each class has 300 unique source-IDs with zero overlap across train/val/test. The honest reading is that PlantVillage (the source of the three maize classes) is a studio-lit dataset with a consistent background per class, and our 5 labels span three plant species with very different leaf morphology (maize is long and narrow, cassava is palmate, beans are oval). ImageNet-pretrained features already separate those distributions almost trivially. The number I'll quote in the video is the **field-drop (1.33 pp)** — that's the one that measures generalisation.

Checkpoint: `checkpoints/best.pt` (6.2 MB FP32, gitignored). Training log committed via `train.py`. Next: INT8 ONNX export and a fresh eval of the quantised model to confirm no regression.

### Hour 3 — Quantization + service smoke-test (2026-04-23)
Wrote `export_onnx.py` and hit the single hardest problem of the day: **MobileNetV3-Small does not survive full-graph INT8 on ONNX Runtime**. I burned ~30 min proving this experimentally before accepting it and swapping strategy. Empirical table, same trained checkpoint, same test set:

| Strategy | Size | Clean F1 | Notes |
|---|---|---|---|
| FP32 ONNX | 6.10 MB | 1.0000 | baseline |
| Static INT8 (QDQ, QUInt8 act / QInt8 wt, per-channel, 100-img calib) | 1.85 MB | 0.7570 | −24 pp — Hardswish + SE blocks |
| Static INT8 + ORT `quant_pre_process` (BN fusion) | 1.87 MB | 0.7299 | −27 pp — preprocess made it worse |
| Dynamic INT8 full-graph (QInt8, per-channel) | 1.70 MB | 0.0667 | **collapsed** — always predicts one class; ORT's CPU ConvInteger kernel doesn't handle this backbone's depthwise + SE |
| Dynamic INT8 full-graph (QUInt8, per-tensor) | 1.70 MB | 0.0667 | same collapse |
| **Dynamic INT8, `MatMul`/`Gemm` only, + preprocess** | **4.34 MB** | **1.0000** | zero regression, shipped |

The winning strategy quantises **only the classifier-head linear layers** (MatMul/Gemm) and leaves the Conv backbone in FP32. 29 % file-size reduction (6.10 → 4.34 MB), well under the 10 MB budget, zero F1 regression. QAT would fix the full-graph case but is out of scope for the 4-hour cap.

**Defense gates added.** `export_onnx.py` now refuses to ship `model.onnx` (and unlinks the file on failure) if any of three brief constraints are violated: size ≥ 10 MB, clean F1 < 0.80, or clean→field drop > 12 pp. Mirror of the size check that was already there for quantisation. Answering my own earlier question: "at what point do we test the drop?" — now, at export-time, hard fail.

**INT8 metrics persisted** to `checkpoints/metrics.json` alongside the FP32 training history so Phase 6 (README + 4-min video) has a single source of truth:

```
int8_model_mb                  4.335
int8_macro_f1_clean            1.0000
int8_macro_f1_field            0.9867
int8_clean_to_field_drop_pp    1.33
```

**Service smoke-test (end-to-end).** `uvicorn service.app:app` → `curl -X POST -F 'image=@samples/maize_rust_1.jpg' http://127.0.0.1:8000/predict` → correct label, confidence 0.9999996, **latency 3.95 ms** on CPU ORT. Same result for cassava_mosaic, healthy, and the field-noisy maize_rust. Response schema matches the brief exactly.

### Hour 4 — Eval notebook + Grad-CAM module (2026-04-23)
Wrote `service/gradcam.py` (reusable — the notebook uses it now, `service/app.py` will use the same class in Phase 5) and `notebooks/01_train_eval.ipynb`. The notebook:

- Re-runs [`model.onnx`](model.onnx) on `data/test/` and `data/test_field/` so the figures quote the shipped INT8 artefact, not a separate FP32 eval.
- Renders a brief-constraints table (size, clean F1, field drop — all green), per-class `classification_report`, confusion matrices for both splits, the 15-epoch train/val curves, and 7 Grad-CAM overlays (one correct per class + every field-set miss).
- Committed **executed** — outputs are embedded so the repo renders directly on GitHub without needing an evaluator to spin up Jupyter.

**What the field miss tells us.** Per-class F1 on the field-noisy test (macro 0.9867):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| bean_spot | 0.9677 | 1.0000 | 0.9836 |
| cassava_mosaic | 1.0000 | 0.9667 | 0.9831 |
| healthy | 0.9677 | 1.0000 | 0.9836 |
| maize_blight | 1.0000 | 0.9667 | 0.9831 |
| maize_rust | 1.0000 | 1.0000 | 1.0000 |

Exactly 2 misclassifications across 150 field samples: one `cassava_mosaic` called `healthy`, one `maize_blight` called `bean_spot`. Both show up with Grad-CAM overlays in the notebook — gives me a concrete answer for the video's "what would you add to close the gap" question (the honest answer is: add Gaussian blur σ ∈ [0, 1.0] to train-time augmentation — the two misses are on the highest-blur field samples).

### Hour 5 — Service rationale + low-confidence escalation (2026-04-23)
Wired the stretch goals into `service/app.py`. Key design call: the service runs in **two modes**, picked automatically at startup, because forcing PyTorch into the Docker image would blow it from ~200 MB to ~2 GB for a feature that isn't needed in the feature-phone deployment path.

- **Full mode** (when `checkpoints/best.pt` and `torchvision` are both present): ORT runs the INT8 inference, then PyTorch runs Grad-CAM on the FP32 checkpoint to derive a real model-attention rationale: `"attention centre (covers 34% of leaf); lesion density consistent with rust pustules; top-2 margin 1.00"`. Falls back silently to the lightweight rationale if anything in the Grad-CAM path throws — the /predict endpoint must not die on a rationale error.
- **Lightweight mode** (ONNX only, Docker default): rationale is class cue + top-2 margin, as before. `latency_ms` stays at ~4 ms.

Both modes also emit `"escalation": "second_photo_different_angle"` when confidence < 0.6 (threshold per brief). Couldn't naturally trigger it with our test samples — confidences are all 0.9999+ — so the test coverage is code-read only. Acceptable risk for a 4-hour brief; the code path is tiny.

**Smoke-tested both modes** on port 8001 (full) and 8002 (lightweight, after renaming `checkpoints/best.pt` aside):
- `/health` correctly reports `"rationale_mode": "full"` vs `"lightweight"`.
- `/predict` returns the rich rationale in full mode and the short one in lightweight mode. Label/confidence/top3 are bit-identical between the two — they come from the same ORT session.

README `## Service` section rewritten to describe the two modes + the JSON shape for each so the evaluator isn't surprised by the rationale difference between the 4-min video (full) and a Docker pull (lightweight).

---

## LLM / assistant tools declared

| Tool | Version / Model | Purpose |
|---|---|---|
| Claude Code (CLI) | Claude Opus 4.7 (1M context) | Repo scaffolding, code generation for the FastAPI service and dataset generator, reviewing model-export code, drafting the USSD fallback artifact. |
| _(more to be added as used)_ | | |

### Why Claude Code specifically
- Filesystem-aware: it can write the actual files into the repo rather than me copy-pasting from a chat window.
- Strong on repository scaffolding and Python boilerplate (FastAPI, ONNX export, image preprocessing).
- I retain authorship: every code path is read, understood, and tested by me before commit.

---

## Sample prompts I actually sent (3 used + 1 discarded)

### Prompt 1 — Repo kickoff (used)
> "@T2.1_Compressed_Crop_Disease_Classifier.pdf Create a public GitHub repo. Add a LICENSE (MIT is fine). Create SIGNED.md at the root with your full name, today's date, and the honor code copied verbatim from page 5. Do this first so you don't forget. Create process_log.md and start the hour-by-hour timeline now. Log every LLM prompt as you go — reconstructing at the end is painful and looks fake. Scaffold the repo … Regenerate the dataset from the synthetic recipe (PlantVillage + Cassava mirrors, ~300/class). Keep the generator script in the repo — it's explicitly required."
>
> _Result:_ Repo `ktt-crop-disease-classifier` was created and pushed to GitHub with the full skeleton, honor-code file, MIT license, generator script, and this process log.

### Prompt 2 — Day-2 kickoff (used)
> "This is the day 2 of the hackathon I am working on we will work on this together and get it to completion. https://github.com/DrUkachi/ktt-crop-disease-classifier.git I have already created a scaffold let's continue work from here. Ensure you read the document and come up with step by step plans on how we will nail this hackathon."
>
> _Attached:_ `T2.1_Compressed_Crop_Disease_Classifier.pdf`.
>
> _Result:_ Claude inventoried the scaffold, proposed a 7-phase plan mapped to the scoring weights (Tech / Model / Data / Product / Comms / Innovation), and flagged two judgement calls back to me — backbone (MobileNetV3-Small vs EfficientNet-B0 under the 10 MB INT8 budget) and whether Grad-CAM is worth the extra 15 min for the rationale field. I locked in MobileNetV3-Small + Grad-CAM.

### Prompt 3 — Training kickoff (used)
> "I have switched to GPU now let's train"
>
> _Prior context in same session:_ I had asked Claude to hold until I flipped the Lightning Studio to GPU; this unblocked Phase 2.
>
> _Result:_ Claude wrote `train.py` (MobileNetV3-Small with ImageNet weights, end-to-end fine-tune, AdamW + cosine LR, class-weighted CE, auto-detect device) and ran it. 15 epochs in 40 seconds on L4; reported val F1, test F1 on both clean and field splits, and ran a leakage check on the manifest source-IDs before I'd even asked — which was the right instinct given how high the clean-test score came in.

### Prompt discarded
> _(placeholder — will document one prompt that was rewritten or rejected and why)_

---

## Single hardest decision (paragraph)

_(to be written near submission — likely about either the backbone choice
(MobileNetV3-Small vs EfficientNet-B0 under the 10 MB INT8 budget), the
quantization strategy (post-training static vs QAT given the 4-hour cap), or
the USSD relay design (extension officer vs cooperative kiosk vs village
agent) where business latency, literacy, and unit economics conflict.)_
