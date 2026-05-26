"""
nilm_explainer.py
=================
Stateless NILM explainability layer.

When a daily window is flagged as anomalous by the IForest, this module
disaggregates the total consumption into per-appliance estimates and
compares them to a baseline (the median of normal windows for that house).

This module does NOT use nilmtk's disaggregation algorithms (which require
labelled training data and complex pipelines).  Instead, it uses a simple
proportional allocation approach based on each appliance's average share
of total consumption across normal days.  This is sufficient for the
"which appliance looks unusual?" explanation goal.

Interface contract
------------------
    explainer = NilmExplainer.build(windows_df)   # call once on normal windows
    explanation = explainer.explain(window_row)    # call per flagged window
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

HOUR_COLS = [f"h{i:02d}" for i in range(24)]
APPLIANCE_COLS = [f"Appliance{i}" for i in range(1, 10)]


class Explanation(NamedTuple):
    """Result of a single-window NILM explanation."""
    date: str
    house_id: int | None
    flagged_total_kwh: float
    baseline_total_kwh: float
    appliance_flagged: dict[str, float]   # appliance → estimated kWh (flagged day)
    appliance_baseline: dict[str, float]  # appliance → baseline kWh
    appliance_delta: dict[str, float]     # appliance → delta (flagged − baseline)
    dominant_anomaly: str                 # appliance with largest absolute delta


class NilmExplainer:
    """Stateless explainer built from normal daily windows.

    Build once from the normal-days slice of the windows dataframe,
    then call ``.explain()`` for any flagged window row.
    """

    def __init__(
        self,
        baseline_totals: dict[int, float],
        appliance_shares: dict[int, dict[str, float]],
        global_baseline: float,
        global_shares: dict[str, float],
        appliance_df: pd.DataFrame | None = None,
        appliance_baselines: dict[int, dict[str, float]] | None = None,
    ) -> None:
        # Per-house averages (keyed by house_id)
        self._baseline_totals = baseline_totals       # house_id → avg daily kWh
        self._appliance_shares = appliance_shares     # house_id → {appliance: share}
        # Global fallback (when house_id not in training data)
        self._global_baseline = global_baseline
        self._global_shares = global_shares
        self._appliance_df = appliance_df
        self._appliance_baselines = appliance_baselines or {}

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        raw_windows_df: pd.DataFrame,
        appliance_df: pd.DataFrame | None = None,
    ) -> "NilmExplainer":
        """Build a NilmExplainer from normal daily windows.

        Parameters
        ----------
        raw_windows_df :
            The full windows DataFrame after attack injection.
            Normal windows (``label == 'normal'``) are used as baseline.
        appliance_df :
            Optional DataFrame with per-appliance hourly readings. If provided
            it must have columns ``house_id``, ``date``, ``Appliance1``..N.
            When not provided, equal shares are assigned (graceful fallback).
        """
        normal = raw_windows_df[raw_windows_df["label"] == "normal"].copy()
        hour_cols = HOUR_COLS

        normal["daily_kwh"] = normal[hour_cols].sum(axis=1) / 1000.0  # W→kWh

        baseline_totals: dict[int, float] = {}
        appliance_shares: dict[int, dict[str, float]] = {}
        appliance_baselines: dict[int, dict[str, float]] = {}

        for house_id, grp in normal.groupby("house_id"):
            baseline_totals[house_id] = float(grp["daily_kwh"].median())

            if appliance_df is not None:
                house_app = appliance_df[appliance_df["house_id"] == house_id]
                app_cols = [c for c in APPLIANCE_COLS if c in house_app.columns]
                if app_cols:
                    shares = {}
                    total = house_app[app_cols].values.sum()
                    for col in app_cols:
                        shares[col] = float(house_app[col].values.sum() / (total + 1e-9))
                    appliance_shares[house_id] = shares

                    # Per-appliance median baseline (across normal days)
                    baselines = {}
                    for col in app_cols:
                        baselines[col] = float(house_app[col].median())
                    appliance_baselines[house_id] = baselines
                    continue

            # Fallback: equal shares across 9 appliances
            n_app = 9
            appliance_shares[house_id] = {
                f"Appliance{i}": 1.0 / n_app for i in range(1, n_app + 1)
            }

        global_baseline = float(normal["daily_kwh"].median())
        global_shares = {f"Appliance{i}": 1.0 / 9 for i in range(1, 10)}

        return cls(
            baseline_totals, appliance_shares, global_baseline, global_shares,
            appliance_df=appliance_df, appliance_baselines=appliance_baselines,
        )

    # ------------------------------------------------------------------
    # Explain a single flagged window
    # ------------------------------------------------------------------

    def explain(self, window_row: pd.Series) -> Explanation:
        """Produce a per-appliance breakdown for one flagged daily window.

        Parameters
        ----------
        window_row :
            A single row from the windows DataFrame (must have ``h00``..``h23``).
        """
        house_id = int(window_row["house_id"]) if "house_id" in window_row else None
        date = str(window_row.get("date", "unknown"))
        readings = window_row[HOUR_COLS].values.astype(np.float64)
        flagged_kwh = float(readings.sum() / 1000.0)

        if house_id is not None and house_id in self._baseline_totals:
            baseline_kwh = self._baseline_totals[house_id]
            shares = self._appliance_shares[house_id]
        else:
            baseline_kwh = self._global_baseline
            shares = self._global_shares

        # Attempt direct appliance row lookup
        app_flagged: dict[str, float] | None = None
        app_baseline: dict[str, float] | None = None
        if (
            self._appliance_df is not None
            and house_id is not None
            and date != "unknown"
        ):
            match = self._appliance_df[
                (self._appliance_df["house_id"] == house_id)
                & (self._appliance_df["date"] == date)
            ]
            if not match.empty:
                row = match.iloc[0]
                app_cols = [c for c in APPLIANCE_COLS if c in row.index]
                if app_cols:
                    app_flagged = {col: float(row[col]) for col in app_cols}
                    if house_id in self._appliance_baselines:
                        app_baseline = self._appliance_baselines[house_id]
                    else:
                        app_baseline = {col: 0.0 for col in app_cols}
                    app_delta = {
                        app: app_flagged[app] - app_baseline[app]
                        for app in app_baseline
                    }
                    dominant = max(app_delta, key=lambda k: abs(app_delta[k] / (app_baseline[k] + 1e-9)))
                    return Explanation(
                        date=date,
                        house_id=house_id,
                        flagged_total_kwh=flagged_kwh,
                        baseline_total_kwh=baseline_kwh,
                        appliance_flagged=app_flagged,
                        appliance_baseline=app_baseline,
                        appliance_delta=app_delta,
                        dominant_anomaly=dominant,
                    )

        # Fallback: proportional allocation
        app_flagged = {app: flagged_kwh * share for app, share in shares.items()}
        app_baseline = {app: baseline_kwh * share for app, share in shares.items()}
        app_delta = {app: app_flagged[app] - app_baseline[app] for app in shares}

        dominant = "unknown"

        return Explanation(
            date=date,
            house_id=house_id,
            flagged_total_kwh=flagged_kwh,
            baseline_total_kwh=baseline_kwh,
            appliance_flagged=app_flagged,
            appliance_baseline=app_baseline,
            appliance_delta=app_delta,
            dominant_anomaly=dominant,
        )

    # ------------------------------------------------------------------
    # Batch explain
    # ------------------------------------------------------------------

    def explain_batch(self, flagged_df: pd.DataFrame) -> list[Explanation]:
        """Explain all flagged rows in a DataFrame."""
        return [self.explain(row) for _, row in flagged_df.iterrows()]
