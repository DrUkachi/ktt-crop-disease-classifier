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

### Hour 2 — TBD
_(quantization to INT8, ONNX export, size verification < 10 MB)_

### Hour 3 — TBD
_(FastAPI service, Dockerfile, sample curl run, robustness eval on field set)_

### Hour 4 — TBD
_(ussd_fallback.md, README polish, video, final push)_

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

### Prompt 3 — _(to fill in: quantization prompt)_
> _(placeholder for the next prompt — will be added when sent)_

### Prompt discarded
> _(placeholder — will document one prompt that was rewritten or rejected and why)_

---

## Single hardest decision (paragraph)

_(to be written near submission — likely about either the backbone choice
(MobileNetV3-Small vs EfficientNet-B0 under the 10 MB INT8 budget), the
quantization strategy (post-training static vs QAT given the 4-hour cap), or
the USSD relay design (extension officer vs cooperative kiosk vs village
agent) where business latency, literacy, and unit economics conflict.)_
