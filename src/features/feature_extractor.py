"""
feature_extractor.py
====================
Extracts a fixed-size feature vector from a single daily window (24 hourly
readings) to feed into the IForest anomaly detector.

Two methods are available (selected at CLI via ``--features``):

    ``engineered`` (default)
        14-dimensional hand-crafted feature vector targeting the 6 attack types.

    ``raw``
        24-dimensional vector — the hourly readings themselves, standardised
        to zero-mean / unit-variance per window before fitting.

Feature groups and rationale:

    Group A — Statistical summaries
        mean, median, min, max, std
        → baseline distribution shape; h1/h3/h4/h5 all shift these

    Group B — Shape descriptors
        peak_to_avg_ratio, skewness, kurtosis
        → h5 (flat line) collapses skewness/kurtosis; h6 (reversal) shifts peak_time

    Group C — Temporal smoothness
        mean_abs_diff (MAD)
        → h3/h4 inject per-step randomness; h2 creates step discontinuities

    Group D — Sparsity
        zero_ratio  (fraction of hours with reading == 0)
        → h2 (zero-window) directly inflates this

    Group E — Time-block energy
        sum of 4 × 6-hour blocks: night (00-05), morning (06-11),
        afternoon (12-17), evening (18-23)
        → h6 reversal swaps night/evening energy signatures;
           h1/h2/h3 reduce block totals disproportionately

Total dimensionality: 5 + 3 + 1 + 1 + 4 = 14 features.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HOUR_COLS = [f"h{i:02d}" for i in range(24)]

FEATURE_NAMES: list[str] = [
    # A
    "mean",
    "median",
    "min",
    "max",
    "std",
    # B
    "peak_to_avg_ratio",
    "skewness",
    "kurtosis",
    # C
    "mean_abs_diff",
    # D
    "zero_ratio",
    # E
    "block_night",
    "block_morning",
    "block_afternoon",
    "block_evening",
]

# Raw method: normalised 24-hour profile + original mean + std = 26 features
FEATURE_NAMES_RAW: list[str] = [
    *[f"{h}_norm" for h in HOUR_COLS],  # h00_norm .. h23_norm
    "raw_mean",
    "raw_std",
]

FEATURE_METHOD_ENGINEERED = "engineered"
FEATURE_METHOD_RAW = "raw"

_EPS = 1e-9  # avoid divide-by-zero


# ---------------------------------------------------------------------------
# Core extractor
# ---------------------------------------------------------------------------


def extract_daily_features(x: np.ndarray) -> np.ndarray:
    """Return a 14-dimensional feature vector for one day's hourly readings.

    Parameters
    ----------
    x : array of shape (24,)
        Hourly aggregate consumption readings (Watts or Wh).

    Returns
    -------
    np.ndarray of shape (14,), dtype float64
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (24,):
        raise ValueError(f"Expected shape (24,), got {x.shape}")

    # --- Group A ---
    feat_mean = np.mean(x)
    feat_median = np.median(x)
    feat_min = np.min(x)
    feat_max = np.max(x)
    feat_std = np.std(x)

    # --- Group B ---
    feat_par = feat_max / (feat_mean + _EPS)
    # Guard against catastrophic cancellation on near-constant windows (e.g. h5 attack)
    if feat_std < _EPS:
        feat_skew = 0.0
        feat_kurt = 0.0
    else:
        feat_skew = float(stats.skew(x))
        feat_kurt = float(stats.kurtosis(x))

    # --- Group C ---
    feat_mad = np.mean(np.abs(np.diff(x)))

    # --- Group D ---
    feat_zero_ratio = float(np.sum(x == 0.0)) / 24.0

    # --- Group E (6-hour blocks) ---
    feat_night = np.sum(x[0:6])
    feat_morning = np.sum(x[6:12])
    feat_afternoon = np.sum(x[12:18])
    feat_evening = np.sum(x[18:24])

    return np.array(
        [
            feat_mean,
            feat_median,
            feat_min,
            feat_max,
            feat_std,
            feat_par,
            feat_skew,
            feat_kurt,
            feat_mad,
            feat_zero_ratio,
            feat_night,
            feat_morning,
            feat_afternoon,
            feat_evening,
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Raw feature extractor
# ---------------------------------------------------------------------------


def extract_raw_features(x: np.ndarray) -> np.ndarray:
    """Return a 26-dimensional raw feature vector for one day's hourly readings.

    The 24-hour series is normalised to zero-mean / unit-variance so that
    IForest sees profile *shape* independently of absolute consumption level.
    The original mean and std are then appended so scale information is still
    available to detect constant-scaling attacks (h1, h3, h4).

    Layout: [h00_norm, ..., h23_norm, raw_mean, raw_std]  — shape (26,)
    """
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (24,):
        raise ValueError(f"Expected shape (24,), got {x.shape}")

    raw_mean = np.mean(x)
    raw_std = np.std(x)

    # Normalise; guard against zero-std (h5 attack produces a flat line)
    if raw_std < _EPS:
        x_norm = np.zeros(24, dtype=np.float64)
    else:
        x_norm = (x - raw_mean) / raw_std

    return np.concatenate([x_norm, [raw_mean, raw_std]])


# ---------------------------------------------------------------------------
# Batch extraction
# ---------------------------------------------------------------------------


def extract_features_from_df(
    windows_df: pd.DataFrame,
    method: str = FEATURE_METHOD_ENGINEERED,
) -> pd.DataFrame:
    """Apply feature extraction to every row in a windows DataFrame.

    Parameters
    ----------
    windows_df :
        DataFrame with columns ``h00``..``h23``, plus any metadata columns.
    method :
        ``"engineered"`` (default) — 14-dim hand-crafted features.
        ``"raw"``                  — 24-dim hourly readings passed directly.

    Returns
    -------
    DataFrame with feature columns, plus ``house_id``, ``date``, ``label``,
    ``attacked`` carried through if present.
    """
    if method == FEATURE_METHOD_RAW:
        extractor_fn = extract_raw_features
        feature_cols = FEATURE_NAMES_RAW
    elif method == FEATURE_METHOD_ENGINEERED:
        extractor_fn = extract_daily_features
        feature_cols = FEATURE_NAMES
    else:
        raise ValueError(f"Unknown feature method: {method!r}. Choose 'engineered' or 'raw'.")

    X = windows_df[HOUR_COLS].values  # (N, 24)
    features = np.apply_along_axis(extractor_fn, axis=1, arr=X)

    feat_df = pd.DataFrame(features, columns=feature_cols, index=windows_df.index)

    # Carry through metadata columns if present
    for meta_col in ["house_id", "date", "label", "attacked"]:
        if meta_col in windows_df.columns:
            feat_df[meta_col] = windows_df[meta_col].values

    # Drop any rows with non-finite values (can occur on degenerate windows)
    n_before = len(feat_df)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    n_dropped = n_before - len(feat_df)
    if n_dropped:
        logger.warning("Dropped %d rows with non-finite features.", n_dropped)

    logger.info("Feature extraction complete: method=%s, shape=%s", method, feat_df[feature_cols].shape)
    return feat_df
