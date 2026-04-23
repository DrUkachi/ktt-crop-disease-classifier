# USSD / Feature-Phone Fallback — Product & Business Artefact

**Audience:** Rwandan smallholder maize / cassava / bean farmers, including
those with a feature phone only (no smartphone, intermittent power, often
multilingual: Kinyarwanda first, French second, English third).

**Constraint reminder from the brief:** the service cannot accept images over
SMS. So the image has to reach the model through a human or kiosk relay, but
the **diagnosis** has to come back through a channel the farmer already has
(USSD prompt + SMS).

---

## 3-step relay workflow

| # | Step                | Actor                              | Channel                | Latency target |
|---|---------------------|------------------------------------|------------------------|----------------|
| 1 | **Photo capture**   | Village agent OR cooperative kiosk | Android phone / tablet | < 2 min        |
| 2 | **Upload + diagnose** | Village agent → FastAPI `/predict` over 3G/Wi-Fi | HTTP        | < 5 s server-side |
| 3 | **Diagnosis delivery** | USSD push or SMS to farmer's MSISDN | `*162#` short-code or SMS | < 60 s end-to-end |

### Step 1 — Photo capture
Two parallel paths so coverage doesn't depend on a single point of failure:

- **(a) Cooperative kiosk:** weekly "diagnosis day" at the cooperative
  building. Farmers bring a leaf, the kiosk operator captures with a tablet on
  a fixed mount and even lighting. Throughput ≈ 60 farmers / day.
- **(b) Village agent (umugozi w'umuhinzi):** roving agent on a motorbike
  visits 8–10 farms / day, takes photos in-field with a low-end Android phone.
  Captures the farmer's MSISDN and a 4-digit plot ID at the same time.

### Step 2 — Upload + diagnose
- Agent's phone runs a thin PWA that POSTs the JPEG to `/predict`.
- Offline-tolerant: PWA queues uploads in IndexedDB and drains when 3G is back.
- Response carries `{ label, confidence, top3, rationale }`. If
  `confidence < 0.6`, the PWA prompts the agent to take a second photo from a
  different angle (low-confidence escalation — see brief stretch goal).

### Step 3 — Diagnosis delivery
- Backend pushes the diagnosis to the farmer via:
  - **USSD callback** on `*162*7#` (preferred — interactive, free to the farmer
    on most Rwandan MNOs); or
  - **SMS** if USSD callback fails (charged to the cooperative, ~RWF 12 / msg).

---

## SMS / USSD diagnosis template

The same payload is rendered in **Kinyarwanda first**, then French, then a
short English tail (some buyers / extension officers use English).

### After a `maize_rust` detection, confidence ≥ 0.85

```
Diagnose: IBARA RY'IBINYABUTAKA ku bigori byawe (maize rust).
Gerageza: kuvanga 2g sulfate y'umuringa muri litiro 1 y'amazi,
ushyire ku mababi inshuro 2 mu cyumweru.
Ibyemezo: hagarara guhinga ibigori aho hamwe iminsi 14.

FR: Diagnostic: rouille du maïs. Mélanger 2g de sulfate de cuivre
dans 1L d'eau et pulvériser 2x/semaine. Ne pas replanter au même
endroit pendant 14 jours.

EN: maize rust, confidence 0.93. Spray 2g copper sulfate / 1L water,
twice a week. Avoid replanting on same plot for 14 days.

Hamagara umufasha (Call agent): *162*7*1#
```

### Word-order / language defence (one paragraph)

Kinyarwanda comes first because it is the L1 of ~99% of rural smallholders;
putting the French or English line first costs trust and comprehension and
risks the message being ignored. Inside the Kinyarwanda block I lead with the
**diagnosis noun** ("Ibara ry'ibinyabutaka") rather than the verb, because
SMS preview windows on feature phones often show only the first ~40
characters; a noun-led lede ("Rust on your maize") is parseable from the
preview alone, while a verb-led lede ("We have detected…") wastes the preview
on filler. Numeric dosing ("2g / 1L") is given before the schedule so a
literate neighbour reading the SMS aloud can act on the most decision-relevant
piece first. The action-recall short-code at the end (`*162*7*1#`) is a
single-tap escalation back to a human, which is non-negotiable for low-trust
adoption.

---

## Unit economics — 1,000 farmers, weekly cooperative kiosk

| Line item                            | Cost (RWF) | Notes |
|--------------------------------------|------------|-------|
| Kiosk operator (1 day/wk × 4 wk)     | 40,000     | RWF 10,000 / day stipend |
| Tablet amortization                  | 4,200      | RWF 200,000 over 48 months |
| 3G data, 1,000 uploads × 200 KB      | 6,000      | Bulk MTN / Airtel B2B rate |
| USSD push (free) + SMS fallback 30%  | 3,600      | 300 fallbacks × RWF 12 |
| Cloud inference (CPU, ONNX)          | 1,500      | ~RWF 0.0015 / inference @ free-tier overflow |
| **Total monthly**                    | **55,300** | |
| **Diagnoses / month**                | **1,000**  | once per farmer |
| **Cost per diagnosis**               | **≈ RWF 55** | |

### Break-even crop-value threshold

A maize-rust outbreak on a 0.25 ha smallholding can wipe out roughly
30–50% of yield if untreated. Average smallholder yield ≈ 600 kg / 0.25 ha,
farm-gate price ≈ RWF 350 / kg → ~RWF 210,000 of crop value at risk per
farmer per season. A correct, timely diagnosis that enables a RWF 800
copper-sulfate intervention to save even **5%** of that yield returns
~RWF 10,500 to the farmer for a RWF 55 spend — a **~190× return**.

**Break-even threshold:** the service pays for itself if a single diagnosis
across the cohort prevents ≥ RWF 55,000 of crop loss per month — i.e. one
saved 0.07 ha plot every month is enough to fund the entire cohort.

---

## Failure modes and mitigations

| Failure                                 | Mitigation                                              |
|-----------------------------------------|---------------------------------------------------------|
| Low-confidence prediction (`< 0.6`)     | PWA prompts second photo, different angle               |
| Off-class leaf (e.g. tomato)            | Reject + SMS: "Leaf type not recognised, see agent"     |
| MSISDN typo                             | USSD echo-back of last 4 digits before sending          |
| SMS gateway down                        | Diagnosis printed at kiosk + handed on paper            |
| Farmer illiterate                       | Voice IVR variant in Kinyarwanda from same `*162*7#`    |

---

## Open questions for evaluators

- Is MTN Rwanda's USSD short-code allocation policy current? (As of 2026 the
  process for `*162*N#` aggregator codes still goes through RURA.)
- Should the cooperative carry the SMS cost or pass it through? Current
  assumption is cooperative-funded (≈ RWF 3.6 / farmer / month).
