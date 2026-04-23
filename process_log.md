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

### Hour 3 — TBD
_(ONNX export + post-training static INT8, size check < 10 MB, service smoke-test with the sample JPEGs)_

### Hour 4 — TBD
_(eval notebook with confusion matrix + Grad-CAM rationale, HF Hub push, README metrics, 4-min video, final submission checklist)_

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
