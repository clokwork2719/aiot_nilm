## Project Plan: AIoT Electricity Theft Detection

### Pipeline Logic (worth having clear upfront)

```
Sub-metered dataset
    → aggregate to simulate smart meter readings
    → inject synthetic attacks (labelled)
    → feature extraction on aggregate stream
    → IForest anomaly detection
         → if flagged: NILM disaggregation (explainability)
    → Streamlit dashboard (streaming replay)
```

NILM never touches IForest. It only activates on flagged windows. This keeps the two modules cleanly separated.

---

### Phase 1 — Data Preparation
- Load REFIT dataset via NILMTK
- Aggregate sub-metered appliance channels → single "meter" reading, simulating what a smart meter would see
- Apply synthetic attack injection to a subset of households/windows (per your literature methods)
- Output: a labelled dataframe of timestamped aggregate consumption, with attack type annotations

---

### Phase 2 — Feature Engineering
TODO. The final model should work on the daily scale, i.e. it flags whether a day contained anomalous consumption.
There are 6 possible attack types h1 - h6, definitions in possible_daily_attacks.md. 

---

### Phase 3 — Anomaly Detection (pyod IForest)
- Train IForest on windows labelled as *normal only*
- Run inference on full dataset (normal + injected attacks)
- Evaluate against synthetic labels: precision, recall, F1, AUC-ROC

---

### Phase 4 — NILM Explainability Layer
- On any window flagged as anomalous, run NILMTK disaggregation
- Output: per-appliance consumption breakdown for that window vs. a normal baseline
- The "explanation" is simply: *which appliances look unusual compared to normal behaviour*
- Keep this module stateless — it takes a window in, returns a breakdown out

---

### Phase 5 — Streamlit Dashboard
Three views:

| View | Content |
|---|---|
| **Live stream** | Scrolling aggregate consumption chart, anomaly flags overlaid in real time (simulated replay) |
| **Alert detail** | On flag: NILM breakdown chart comparing flagged window vs normal baseline |
| **Summary stats** | Precision/recall, anomaly rate, attack type distribution |

Replay speed should be configurable — useful for demos.

---

### Evaluation Section (for the report)
- Detection metrics per attack type (not just aggregate) — shows the model isn't just catching easy cases
- Contamination sensitivity analysis
- Brief qualitative assessment of NILM explanations: do they point at the right appliances for each attack type?

---

### Suggested Order of Work
1. Get dataset loading + aggregation working in NILMTK first — this is the most likely place to lose time
2. Attack injection + feature engineering
3. IForest pipeline + evaluation
4. NILM explainability module
5. Streamlit last — it's the easiest part and shouldn't block anything else