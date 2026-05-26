"""
data_loader.py
==============
Loads the REFIT dataset from raw CSVs, aggregates appliance channels,
and chunks the signal into hourly-resampled daily windows.

REFIT CSV format (per house):
  Time, Unix, Aggregate, Appliance1, …, ApplianceN, Issues
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APPLIANCE_COLS = [f"Appliance{i}" for i in range(1, 10)]  # up to 9 sub-meters
REFIT_DIR = Path(__file__).parent.parent.parent / "REFIT"
DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def load_house_csv(house_id: int, refit_dir: Path = REFIT_DIR) -> pd.DataFrame:
    """Read a single REFIT house CSV and return a clean DataFrame.

    Columns returned: ``Aggregate``, ``Appliance1``..``ApplianceN``
    Index: DatetimeIndex (UTC-naive, as-recorded).
    """
    path = refit_dir / f"CLEAN_House{house_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"REFIT CSV not found: {path}")

    df = pd.read_csv(
        path,
        usecols=["Time", "Aggregate"] + APPLIANCE_COLS,
        parse_dates=["Time"],
        index_col="Time",
        na_values=["", "NA"],
    )
    # Drop appliance cols that are entirely NaN (some houses have fewer meters)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="any")
    return df


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a high-frequency REFIT DataFrame to 1-hour mean intervals."""
    return df.resample("1h").mean()


def aggregate_to_main_meter(df: pd.DataFrame) -> pd.Series:
    """Return the ``Aggregate`` column (already the summed main meter reading)."""
    return df["Aggregate"]


def get_appliance_columns(df: pd.DataFrame) -> list[str]:
    """Return the appliance columns present in this DataFrame."""
    return [c for c in df.columns if c.startswith("Appliance")]


# ---------------------------------------------------------------------------
# Daily windowing
# ---------------------------------------------------------------------------


def split_into_daily_windows(
    series: pd.Series, min_hours: int = 20
) -> list[tuple[str, np.ndarray]]:
    """Split a 1-h resampled series into (date_str, 24-length array) tuples.

    Days with fewer than ``min_hours`` valid readings are dropped.

    Returns
    -------
    list of (date_str, array_of_24_floats)
    """
    windows: list[tuple[str, np.ndarray]] = []
    for date, group in series.groupby(series.index.date):
        if len(group) < min_hours:
            logger.debug("Skipping %s — only %d hours", date, len(group))
            continue
        # Reindex to exactly 24 hours; forward-fill at most 1 gap
        hour_range = pd.date_range(
            start=f"{date} 00:00", periods=24, freq="1h"
        )
        reindexed = group.reindex(hour_range).ffill().bfill()
        if reindexed.isna().any():
            continue
        windows.append((str(date), reindexed.values.astype(np.float64)))
    return windows


# ---------------------------------------------------------------------------
# High-level pipeline entry point
# ---------------------------------------------------------------------------


def load_all_houses(
    house_ids: list[int] | None = None,
    refit_dir: Path = REFIT_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, resample, and concatenate multiple REFIT houses.

    Returns a tuple of (windows_df, appliance_df).
    windows_df has columns:
        ``house_id``, ``date``, ``h00``..``h23``  (24 hourly readings)
    appliance_df has columns:
        ``house_id``, ``date``, ``Appliance1``..``Appliance9`` (daily sums)
    """
    if house_ids is None:
        # Default to all houses present in REFIT dir
        house_ids = sorted(
            int(p.stem.replace("CLEAN_House", ""))
            for p in refit_dir.glob("CLEAN_House*.csv")
        )

    records: list[dict] = []
    appliance_records: list[dict] = []
    for hid in house_ids:
        logger.info("Loading house %d …", hid)
        try:
            df_raw = load_house_csv(hid, refit_dir)
        except FileNotFoundError:
            logger.warning("House %d CSV not found — skipping.", hid)
            continue

        hourly = resample_hourly(df_raw)
        series = aggregate_to_main_meter(hourly)
        windows = split_into_daily_windows(series)

        # Extract appliance columns present in the house DataFrame
        app_cols = get_appliance_columns(hourly)
        # Group by date and calculate daily sum for appliances
        daily_appliance_sums = hourly[app_cols].groupby(hourly.index.date).sum()

        for date_str, arr in windows:
            rec = {"house_id": hid, "date": date_str}
            rec.update({f"h{i:02d}": arr[i] for i in range(24)})
            records.append(rec)

            # Match daily appliance sums
            dt_date = pd.to_datetime(date_str).date()
            if dt_date in daily_appliance_sums.index:
                app_row = daily_appliance_sums.loc[dt_date]
                app_rec = {"house_id": hid, "date": date_str}
                for col in APPLIANCE_COLS:
                    if col in app_row:
                        app_rec[col] = float(app_row[col])
                    else:
                        app_rec[col] = 0.0
                appliance_records.append(app_rec)

        logger.info("  → %d daily windows from house %d", len(windows), hid)

    if not records:
        raise RuntimeError("No daily windows could be extracted. Check REFIT_DIR.")

    return pd.DataFrame(records), pd.DataFrame(appliance_records)


def save_windows(df: pd.DataFrame, path: Path | str | None = None) -> Path:
    """Persist the daily-windows DataFrame to Parquet."""
    if path is None:
        DATA_DIR.mkdir(exist_ok=True)
        path = DATA_DIR / "daily_windows_raw.parquet"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Saved %d rows → %s", len(df), path)
    return path


def load_windows(path: Path | str | None = None) -> pd.DataFrame:
    """Load the daily-windows DataFrame from Parquet."""
    if path is None:
        path = DATA_DIR / "daily_windows_raw.parquet"
    return pd.read_parquet(path)
