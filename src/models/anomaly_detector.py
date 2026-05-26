"""
anomaly_detector.py
===================
Isolation Forest wrapper for electricity theft detection.

Design contract
---------------
- Training ONLY on *normal* windows (unsupervised anomaly detection).
- Inference produces binary flags (0 = normal, 1 = anomaly) and raw scores.
- Evaluation produces per-attack-type metrics so we can see the model isn't
  just catching easy cases.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from pyod.models.iforest import IForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent.parent / "models"

# Metadata columns that are never features
_META_COLS = {"house_id", "date", "label", "attacked", "pred_flag", "anomaly_score"}

# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class TheftDetector:
    """Isolation Forest wrapper for daily-window theft detection."""

    def __init__(
        self,
        contamination: float = 0.20,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.contamination = contamination
        self.model = IForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
        )
        self._fitted = False
        self._feature_cols: list[str] = []  # set at train time

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, feat_df: pd.DataFrame) -> None:
        """Fit IForest on *normal* windows only.

        Parameters
        ----------
        feat_df :
            Features DataFrame (from ``extract_features_from_df``).
            Must contain a ``label`` column.  Only rows where
            ``label == 'normal'`` are used for fitting.
            Feature columns are inferred by excluding known metadata columns.
        """
        self._feature_cols = [c for c in feat_df.columns if c not in _META_COLS]
        normal_mask = feat_df["label"] == "normal"
        X_train = feat_df.loc[normal_mask, self._feature_cols].values
        logger.info(
            "Training IForest on %d normal windows, %d features …",
            len(X_train), len(self._feature_cols),
        )
        self.model.fit(X_train)
        self._fitted = True
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Run inference on a features DataFrame.

        Returns the input DataFrame with two new columns:
            ``pred_flag``  — 1 if anomaly, 0 if normal
            ``anomaly_score`` — raw IForest anomaly score (higher = more anomalous)
        """
        if not self._fitted:
            raise RuntimeError("Model has not been trained yet. Call .train() first.")

        X = feat_df[self._feature_cols].values
        flags = self.model.predict(X)
        scores = self.model.decision_function(X)  # higher = more anomalous

        out = feat_df.copy()
        out["pred_flag"] = flags
        out["anomaly_score"] = scores
        return out

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, result_df: pd.DataFrame) -> dict:
        """Compute metrics from a DataFrame that has both ``label`` and ``pred_flag``.

        Returns a dict with:
            ``overall``   — classification_report dict for binary detection
            ``per_attack`` — per-attack-type precision/recall/F1
            ``auc_roc``   — overall binary AUC-ROC
            ``confusion_matrix`` — numpy array (binary: normal vs anomaly)
            ``feature_cols`` — list of feature column names used
        """
        df = result_df.copy()
        y_true_binary = (df["label"] != "normal").astype(int)
        y_pred_binary = df["pred_flag"].astype(int)
        y_score = df["anomaly_score"].values

        overall = classification_report(
            y_true_binary, y_pred_binary, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        auc = roc_auc_score(y_true_binary, y_score)

        # Per-attack-type breakdown
        per_attack: dict[str, dict] = {}
        for attack in df["label"].unique():
            if attack == "normal":
                continue
            mask = (df["label"] == attack) | (df["label"] == "normal")
            sub = df[mask]
            y_t = (sub["label"] != "normal").astype(int)
            y_p = sub["pred_flag"].astype(int)
            rep = classification_report(y_t, y_p, output_dict=True, zero_division=0)
            per_attack[attack] = rep.get("1", {})

        metrics = {
            "overall": overall,
            "per_attack": per_attack,
            "auc_roc": auc,
            "confusion_matrix": cm,
            "feature_cols": self._feature_cols,
        }
        logger.info("AUC-ROC: %.4f | features: %s (%d)", auc, self._feature_cols[:3], len(self._feature_cols))
        for k, v in per_attack.items():
            logger.info(
                "%s → precision=%.3f recall=%.3f f1=%.3f",
                k,
                v.get("precision", 0),
                v.get("recall", 0),
                v.get("f1-score", 0),
            )
        return metrics

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str | None = None) -> Path:
        if path is None:
            path = MODEL_DIR / "iforest.pkl"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Model saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Path | str | None = None) -> "TheftDetector":
        if path is None:
            path = MODEL_DIR / "iforest.pkl"
        path = Path(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("Model loaded ← %s", path)
        return obj


# ---------------------------------------------------------------------------
# Standalone evaluation (works without a fitted model instance)
# ---------------------------------------------------------------------------


def evaluate_results(result_df: pd.DataFrame, feature_cols: list[str] | None = None) -> dict:
    """Compute detection metrics from a DataFrame with ``label``, ``pred_flag``,
    and ``anomaly_score`` columns.  Can be called without a fitted TheftDetector.

    Parameters
    ----------
    result_df :
        DataFrame containing at least ``label``, ``pred_flag``, ``anomaly_score``.
    feature_cols :
        Optional list of feature column names to record in the returned metrics dict.

    Returns
    -------
    dict with keys: ``overall``, ``per_attack``, ``auc_roc``,
    ``confusion_matrix``, ``feature_cols``.
    """
    df = result_df.copy()
    y_true_binary = (df["label"] != "normal").astype(int)
    y_pred_binary = df["pred_flag"].astype(int)
    y_score = df["anomaly_score"].values

    overall = classification_report(
        y_true_binary, y_pred_binary, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    auc = roc_auc_score(y_true_binary, y_score)

    per_attack: dict[str, dict] = {}
    for attack in df["label"].unique():
        if attack == "normal":
            continue
        mask = (df["label"] == attack) | (df["label"] == "normal")
        sub = df[mask]
        y_t = (sub["label"] != "normal").astype(int)
        y_p = sub["pred_flag"].astype(int)
        rep = classification_report(y_t, y_p, output_dict=True, zero_division=0)
        per_attack[attack] = rep.get("1", {})

    metrics = {
        "overall": overall,
        "per_attack": per_attack,
        "auc_roc": auc,
        "confusion_matrix": cm,
        "feature_cols": feature_cols or [],
    }
    logger.info("AUC-ROC: %.4f", auc)
    for k, v in per_attack.items():
        logger.info(
            "%s → precision=%.3f recall=%.3f f1=%.3f",
            k, v.get("precision", 0), v.get("recall", 0), v.get("f1-score", 0),
        )
    return metrics
