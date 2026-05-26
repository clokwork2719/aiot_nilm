**NILM Explainer — Fix Implementation Plan**

---

**Problem 1: `appliance_df` is not stored on the instance**

`build()` uses `appliance_df` only to compute shares, then discards it. `explain()` has no access to per-day appliance readings.

Fix: add `appliance_df` as an instance attribute, passed through `__init__` and stored as `self._appliance_df`.

---

**Problem 2: Per-appliance baselines are not precomputed**

The baseline for each appliance (median daily kWh across normal days, per house) is never stored. Direct lookup needs something to compare against.

Fix: during `build()`, for each house compute `median per appliance column` across normal days in `appliance_df`. Store as `self._appliance_baselines: dict[int, dict[str, float]]`.

---

**Problem 3: `explain()` uses proportional allocation as primary path**

Proportional allocation scales all appliances uniformly with total consumption — it is uninformative and should only run when a real lookup is impossible.

Fix: in `explain()`, before proportional allocation, attempt a direct row lookup:
```
appliance_df WHERE house_id == window_row.house_id AND date == window_row.date
```
If a row is found, use its appliance column values directly as `app_flagged`. Use `self._appliance_baselines[house_id]` as `app_baseline`. Fall through to proportional allocation only if the lookup returns empty.

---

**Problem 4: `dominant_anomaly` is misleading under proportional allocation**

Under proportional allocation, the dominant appliance is always the one with the largest historical share, regardless of what actually changed. This is a misleading output.

Fix: when falling back to proportional allocation (no date match), set `dominant_anomaly` to `"unknown"` rather than returning a spurious appliance name.

---

**Scope boundary — do not change:**
- `Explanation` NamedTuple fields
- `explain_batch()`
- The `HOUR_COLS` / `APPLIANCE_COLS` constants
- The `baseline_totals` logic (median daily kWh per house from `windows_df`)