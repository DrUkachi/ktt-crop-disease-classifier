# 4-Minute Video — Shot List & Preparation Notes

Maps to the brief's required segments (pp. 3–4). This is a **prep list, not a
teleprompter** — the brief explicitly warns against polished narration
("show us the work and the thinking, not a polished script"). Speak in your own
voice; the bullets below are the facts to hit in order.

**Total time:** 4:00 hard cap. Budget slightly under each segment to leave
clock-slack for transitions.

---

## Pre-record checklist

- [ ] Camera on, framed for the 0:00–0:30 intro and the 0:50 "this is me" beat in segment 2 if useful.
- [ ] Studio is on **GPU** off, **CPU** on — inference demo is CPU-only and the latency should show that.
- [ ] `uvicorn service.app:app --host 0.0.0.0 --port 8000` running in a terminal. `/health` returns `"rationale_mode": "full"` before recording.
- [ ] Second terminal ready at the repo root, with history cleared (`clear`) and shell prompt visible.
- [ ] VS Code open to [`service/app.py`](service/app.py), scrolled to `_preprocess` / `_rationale_full` / `predict`.
- [ ] [`ussd_fallback.md`](ussd_fallback.md) open in a preview pane (VS Code Markdown preview or the GitHub tab — whichever renders cleaner).
- [ ] Screen resolution set so terminal text is readable on mobile playback (≥ 14 pt).
- [ ] Cheat sheet for segment 5 (the three answers) on a sticky note, out of frame. Numbers below.

---

## 0:00 – 0:30  On-camera intro

Required by brief. On-camera mandatory; face visible.

Beats to hit:
- Full name: **Ukachi Osisiogu**.
- Challenge ID: **T2.1 — Compressed Crop Disease Classifier**, AIMS KTT Hackathon Tier 2.
- Final model size: **4.34 MB** INT8 ONNX. Budget was < 10 MB.
- Quick thesis: "Five crop classes — maize rust, maize blight, cassava mosaic, bean spot, and healthy — classified on-device, wrapped in a FastAPI service, with a feature-phone delivery path for farmers without a smartphone."

~25 s spoken, ~5 s transition into screen share.

---

## 0:30 – 1:30  Size + quantization choices

Live screen-share. Terminal visible, face off-screen is fine.

Commands to run:
```
ls -la model.onnx
du -h model.onnx
```
Expected: `4335078` bytes / `4.2M`. Read the size aloud.

Talking points (in this order):
1. "MobileNetV3-Small pretrained on ImageNet, fine-tuned end-to-end for our 5 classes."
2. "FP32 export is 6.10 MB — already under 10 MB, but I wanted real quantization."
3. **The honest bit**: "I tried static INT8 on the full graph first — clean macro-F1 dropped from 1.0 to 0.73. MobileNetV3's Hardswish plus Squeeze-and-Excitation blocks don't quantise cleanly without QAT."
4. "Then I tried dynamic INT8 on the full graph — collapsed to 0.07, always predicting one class."
5. "**Shipped strategy**: dynamic INT8 on the MatMul and Gemm nodes only — the classifier head. Backbone stays FP32. 29% size reduction, zero accuracy regression."
6. Open `export_onnx.py` in VS Code, scroll to `_quantize_int8`, point at the `op_types_to_quantize=["MatMul", "Gemm"]` line.
7. "Export-time asserts unlink `model.onnx` if any of the three brief budgets regress — size, clean F1, or field drop."

---

## 1:30 – 2:30  Live `/predict` demo

Terminal, split with the JSON response visible.

Commands:

```bash
curl -X POST -F 'image=@samples/maize_rust_1.jpg' http://localhost:8000/predict | jq
```

Read the response aloud, especially:
- `"label": "maize_rust"`
- `"confidence": 0.9999...`
- `"latency_ms": ~4`
- `"rationale": "attention centre (covers 34% of leaf); lesion density consistent with rust pustules; top-2 margin 1.00"` — emphasise "attention centre" is **Grad-CAM derived**, not a lookup table.

Then the field-noisy sample:

```bash
curl -X POST -F 'image=@samples/maize_rust_field_1.jpg' http://localhost:8000/predict | jq
```

Say: "Same leaf, now with blur, JPEG re-compression, and brightness jitter — the brief's field-noisy recipe. Still confident, still correct."

If time: `curl http://localhost:8000/health | jq` to show `rationale_mode: full`, and mention "on Docker it's `lightweight` — the Dockerfile doesn't ship PyTorch".

---

## 2:30 – 3:30  USSD fallback walk-through

Switch to the rendered [`ussd_fallback.md`](ussd_fallback.md).

Beats in order:
1. "Farmer has a feature phone. Service cannot accept images over SMS, per the brief. So the image has to reach the model through a human or kiosk relay, but the diagnosis has to come back on a channel the farmer already has."
2. Scroll to the **3-step relay workflow table**. Read the three rows: photo capture, upload + diagnose, diagnosis delivery. Time targets: < 2 min, < 5 s, < 60 s.
3. Scroll to the **SMS template** — don't read the full template yet, that's in segment 5.
4. Mention: "Two parallel capture paths — weekly cooperative kiosk, OR a roving village agent. Neither is a single point of failure."
5. Scroll to **unit economics** — "RWF 55 per diagnosis, ~190× return when a saved plot's worth RWF 10,500."
6. Point at the **failure-modes table** — "Low confidence, off-class leaf, MSISDN typo, SMS gateway down, illiteracy — each has a mitigation."

Keep this segment tight — 60 s goes fast.

---

## 3:30 – 4:00  Three questions

On-camera. Answer in order, roughly 10 s each.

### Q1 — Technical
*"Your macro-F1 dropped from the clean test set to the field-noisy set. By how many points, and name the single augmentation you would add next to close the gap."*

> **Answer:** "**1.33 percentage points** — from 1.0000 on clean to 0.9867 on field-noisy. The single augmentation I'd add next is **Gaussian blur with sigma in [0, 1.0] at train time**. I deliberately excluded blur from train augmentation so the drop I report is honest — the two field misses in the notebook are both on the highest-sigma field samples, which confirms blur is the dominant robustness axis."

### Q2 — Product/Business
*"A cooperative has 1,000 farmers and agrees to run a weekly kiosk-diagnosis day. Give me the cost per diagnosis and the break-even crop-value threshold, in RWF, on camera."*

> **Answer:** "**RWF 55 per diagnosis** for a 1,000-farmer cohort running a weekly kiosk day — that's kiosk operator, tablet amortization, 3G data, USSD push plus SMS fallback, and cloud inference, totalling about RWF 55,300 per month across 1,000 diagnoses. **Break-even: the service pays for itself if it prevents RWF 55,000 of crop loss per month — roughly one saved 0.07-hectare plot per month across the whole cohort.** On a per-farmer basis, a RWF 800 copper-sulfate intervention saving 5% of a RWF 210,000 plot returns ~RWF 10,500 for a RWF 55 spend, so the expected return is roughly 190×."

### Q3 — Local context
*"Read the SMS diagnosis message a farmer receives after a maize_rust detection. Defend your language and word-order choice."*

> **Read the message first, then the defence.**
>
> *(Read:)* **"Diagnose: Ibara ry'ibinyabutaka ku bigori byawe — maize rust. Gerageza kuvanga 2g sulfate y'umuringa muri litiro 1 y'amazi, ushyire ku mababi inshuro 2 mu cyumweru. Hagarara guhinga ibigori aho hamwe iminsi 14. FR: rouille du maïs, 2g sulfate de cuivre par litre, deux fois par semaine. EN: maize rust, confidence 0.93, spray 2g copper sulfate per litre, twice a week, avoid replanting same plot 14 days. Hamagara umufasha: star-162-star-7-star-1-hash."**
>
> *(Defence:)* "Kinyarwanda leads because it is the L1 of roughly 99% of rural smallholders — French or English first would cost trust. Inside the Kinyarwanda block I lead with the **diagnosis noun**, not a verb, because feature-phone SMS previews show only about the first 40 characters — a noun-led lede is parseable from the preview alone. Numeric dosing — 2g per litre — comes before the treatment schedule, so a literate neighbour reading the SMS aloud can act on the most decision-relevant piece first. The action short-code at the end — star-one-six-two — is a single-tap escalation back to a human, which is non-negotiable for low-trust adoption."

---

## Post-record

- Trim to ≤ 4:00. Brief is strict on cap.
- Upload to YouTube unlisted. Paste URL into README.md "Submission artefacts" block and into the HF Hub model card.
- Leave a copy local in case YouTube takes it down during review.

## Three facts you will be asked for repeatedly; memorize them

- **Model size:** 4.34 MB (INT8 ONNX).
- **Clean → field drop:** 1.33 pp.
- **Cost per diagnosis / break-even:** RWF 55 / RWF 55,000 per month of prevented crop loss.
