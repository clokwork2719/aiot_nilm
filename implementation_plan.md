# AIoT Electricity Theft Detection Implementation Plan

This plan expands on the draft `README.md` and provides a concrete, file-by-file execution strategy for building the AIoT electricity theft detection pipeline. It includes specific strategies for the previously undefined Feature Engineering phase to detect the daily attacks defined in `possible_daily_attacks.md`.

## Proposed Changes

---

### Data Preparation & Attack Injection

This component handles loading the raw REFIT dataset, aggregating appliance channels into a main meter reading, and injecting the 6 synthetic attacks. Time resolution will be **hourly** (24 readings per day).

#### [NEW] `src/data/data_loader.py`
- Function to run dataset conversion using `nilmtk.dataset_converters.convert_refit(input_path, output_filename, format='HDF')` to get the raw CSVs into the required HDF format.
- Functions to load the processed REFIT data using `nilmtk.DataSet`.
- Function to aggregate sub-metered appliances into a total aggregate signal.
- Function to chunk the continuous time-series into discrete daily windows (24 readings per window).

#### [NEW] `src/data/attacks.py`
- Implements the 6 attacks defined in `possible_daily_attacks.md` as Python functions operating on a 1D numpy array (a single day's readings $x$, shape `(24,)`):
  - `attack_h1(x)`: Constant scalar reduction ($\alpha \in [0.1, 0.8]$).
  - `attack_h2(x)`: Zero out readings during a random time window.
  - `attack_h3(x)`: Random scalar reduction per timestep ($\gamma_t \in [0.1, 0.8]$).
  - `attack_h4(x)`: Random scalar multiplied by the daily mean.
  - `attack_h5(x)`: Replace all readings with the daily mean.
  - `attack_h6(x)`: Reverse the daily profile ($x_{24-t}$).
- Function `inject_attacks(daily_windows, contamination_rate)` to randomly apply these attacks to a subset of the daily windows and record the ground truth labels (`normal` or `h1`-`h6`).

---

### Feature Engineering

This component extracts daily aggregate features from the raw time-series windows to feed into the anomaly detection model.

#### [NEW] `src/features/feature_extractor.py`
- Function `extract_daily_features(window)` to convert a raw daily time-series array (24 hourly readings) into a fixed-size feature vector. Proposed features to catch the specific attacks:
  1. **Statistical:** Mean, Median, Min, Max, Standard Deviation.
  2. **Shape/Distribution:** Peak-to-Average Ratio, Skewness, Kurtosis.
  3. **Continuity/Smoothness:** Mean Absolute Difference between consecutive time steps (highly effective for catching $h_3$, $h_4$, $h_5$).
  4. **Sparsity:** Ratio of zero-readings (effective for $h_2$).
  5. **Time-Block Energy:** Sum of energy in 4 blocks (Night: 00-06, Morning: 06-12, Afternoon: 12-18, Evening: 18-24). This is crucial for catching the reversal attack ($h_6$).

---

### Anomaly Detection & NILM Explainability

This component trains the Isolation Forest, evaluates it, and triggers NILM on anomalous windows.

#### [NEW] `src/models/anomaly_detector.py`
- Wrapper around `pyod.models.iforest.IForest`.
- `train_iforest()`: Fits the model ONLY on the extracted features of `normal` daily windows.
- `predict_anomalies()`: Runs inference on mixed (normal + attacked) data, outputting binary anomaly flags and anomaly scores.

#### [NEW] `src/models/nilm_explainer.py`
- Function `run_disaggregation(window_timestamp)`: Invoked only when `predict_anomalies` flags a window.
- Uses `nilmtk` to disaggregate the total consumption of that window into appliance-level estimates.
- Function `compare_to_baseline(disaggregated_window, normal_baseline)`: Computes the difference to highlight *which* appliance is causing the anomaly.

---

### Dashboard and Orchestration

This component ties everything together into a Streamlit UI, using static simulation for live replay.

#### [NEW] `src/app/dashboard.py`
- **Streamlit Application** containing:
  - **Sidebar:** Controls for replay speed, selecting households, and adjusting the `IForest` contamination threshold.
  - **Live Stream View:** A dynamic line chart that updates in a loop (loading from a static pre-computed CSV/DataFrame and simulating streaming using `time.sleep()`), showing aggregate consumption and marking red dots where anomalies are detected.
  - **Alert Detail View:** An expander/modal that appears when an anomaly is flagged, displaying the NILM appliance breakdown (bar chart comparing current vs. baseline).
  - **Metrics View:** Displays global evaluation metrics (Precision, Recall, F1) across the 6 attack types.

#### [MODIFY] `main.py`
- Update the main entry point to act as a CLI orchestration tool.
- Commands: 
  - `uv run main.py prepare-data` (runs Phase 1 & 2)
  - `uv run main.py train` (runs Phase 3)
  - `uv run main.py dashboard` (launches Streamlit)

## Verification Plan

### Automated Tests
- **Unit Tests:** Run tests on `src/data/attacks.py` to ensure $h_1$ through $h_6$ accurately modify a dummy 24-hour array according to the mathematical definitions.
- **Feature Extraction Validation:** Verify that `feature_extractor.py` correctly calculates zero-ratios and smoothness.

### Manual Verification
1. Run the data preparation script and inspect the output dataset locally to ensure labels are distributed correctly.
2. Run the `IForest` model and output a local confusion matrix. Verify that detection metrics for each specific attack type ($h_1$ - $h_6$) are logged, ensuring the model catches all varieties, not just the easiest ones.
3. Start the Streamlit dashboard, set replay speed to a visible rate, and visually confirm that the anomaly flags align with injected attacks. Inspect the NILM breakdown chart when an alert is triggered.
