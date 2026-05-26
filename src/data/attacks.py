"""
attacks.py
==========
Synthetic electricity theft attack functions (h1–h6) and batch injection.

All attack functions take a 1-D numpy array of 24 hourly readings and
return a modified copy.  None of them mutate the input.

Mathematical definitions (from possible_daily_attacks.md):

    h1(x_t) = α · x_t                      α ~ Uniform(0.1, 0.8)
    h2(x_t) = β_t · x_t                    β_t = 0 in [start, end), else 1
    h3(x_t) = γ_t · x_t                    γ_t ~ Uniform(0.1, 0.8) per step
    h4(x_t) = γ_t · mean(x)                γ_t ~ Uniform(0.1, 0.8) per step
    h5(x_t) = mean(x)
    h6(x_t) = x_{24-t}                     (time-reversal)
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

AttackFn = Callable[[np.ndarray, np.random.Generator], np.ndarray]

ATTACK_LABELS = ["h1", "h2", "h3", "h4", "h5", "h6"]


# ---------------------------------------------------------------------------
# Individual attack functions
# ---------------------------------------------------------------------------


def attack_h1(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Constant scale reduction: h1(x_t) = α · x_t, α ~ U(0.1, 0.8)."""
    alpha = rng.uniform(0.1, 0.8)
    return x * alpha


def attack_h2(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Zero-out a contiguous time window: β_t = 0 in [start, end), else 1."""
    n = len(x)
    start = rng.integers(0, n - 1)
    end = rng.integers(start + 1, n + 1)
    out = x.copy()
    out[start:end] = 0.0
    return out


def attack_h3(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Per-timestep random scale: h3(x_t) = γ_t · x_t, γ_t ~ U(0.1, 0.8)."""
    gamma = rng.uniform(0.1, 0.8, size=len(x))
    return x * gamma


def attack_h4(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random scalar × daily mean: h4(x_t) = γ_t · mean(x)."""
    gamma = rng.uniform(0.1, 0.8, size=len(x))
    return gamma * np.mean(x)


def attack_h5(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flat daily mean: h5(x_t) = mean(x)."""
    return np.full_like(x, fill_value=np.mean(x))


def attack_h6(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Time reversal: h6(x_t) = x_{24-t}."""
    return x[::-1].copy()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ATTACK_FNS: dict[str, AttackFn] = {
    "h1": attack_h1,
    "h2": attack_h2,
    "h3": attack_h3,
    "h4": attack_h4,
    "h5": attack_h5,
    "h6": attack_h6,
}


# ---------------------------------------------------------------------------
# Batch injection
# ---------------------------------------------------------------------------


def inject_attacks(
    windows_df: pd.DataFrame,
    contamination: float = 0.20,
    seed: int = 42,
    attack_types: list[str] | None = None,
) -> pd.DataFrame:
    """Randomly apply attacks to a subset of daily windows.

    Parameters
    ----------
    windows_df:
        DataFrame produced by ``data_loader.load_all_houses()``.
        Must have columns ``h00``..``h23``, ``house_id``, ``date``.
    contamination:
        Fraction of rows to attack (0 < contamination < 1).
    seed:
        Random seed for reproducibility.
    attack_types:
        Subset of attack labels to use.  Defaults to all six.

    Returns
    -------
    A new DataFrame identical to ``windows_df`` but with two extra columns:
        ``label`` — ``"normal"`` or one of ``"h1"``..``"h6"``
        ``attacked`` — bool flag for convenience
    """
    if attack_types is None:
        attack_types = ATTACK_LABELS

    fns = {k: ATTACK_FNS[k] for k in attack_types}
    rng = np.random.default_rng(seed)

    df = windows_df.copy()
    hour_cols = [f"h{i:02d}" for i in range(24)]

    n = len(df)
    n_attack = int(round(n * contamination))
    attack_idx = rng.choice(n, size=n_attack, replace=False)

    df["label"] = "normal"
    df["attacked"] = False

    chosen_labels = rng.choice(attack_types, size=n_attack)

    for pos, (row_i, label) in enumerate(zip(attack_idx, chosen_labels)):
        x = df.iloc[row_i][hour_cols].values.astype(np.float64)
        x_attacked = fns[label](x, rng)
        df.iloc[row_i, df.columns.get_indexer(hour_cols)] = x_attacked
        df.at[df.index[row_i], "label"] = label
        df.at[df.index[row_i], "attacked"] = True

    n_per_type = {lbl: (df["label"] == lbl).sum() for lbl in ["normal"] + attack_types}
    logger.info(
        "Injected %d attacks into %d windows (%.1f%%). Distribution: %s",
        n_attack,
        n,
        100 * contamination,
        n_per_type,
    )
    return df
