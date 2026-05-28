"""
supervised_detector.py
======================
XGBoost supervised baseline for electricity theft detection.

Unlike TheftDetector (IForest, unsupervised), this model requires labeled
attack windows at training time.  It is used as a comparison baseline to
demonstrate the label-scarcity advantage of the unsupervised approach.

Interface mirrors TheftDetector so both can be used interchangeably in the
compare pipeline.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

logger = logging.getLogger(__name__)

_META_COLS = {"house_id", "date", "label", "attacked", "pred_flag", "anomaly_score"}


class SupervisedDetector:
    """XGBoost wrapper trained on labeled normal + attack windows."""

    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.random_state = random_state
        self._fitted = False
        self._feature_cols: list[str] = []
        self.model: XGBClassifier | None = None

    def train(self, feat_df: pd.DataFrame, label_ratio: float = 1.0) -> None:
        """Fit XGBoost on labeled windows.

        Parameters
        ----------
        feat_df :
            Features DataFrame with a ``label`` column.
        label_ratio :
            Fraction of attack windows whose labels are revealed during training
            (0 < label_ratio <= 1.0).  Simulates real-world label scarcity:
            at 1.0 XGBoost sees every injected attack; at 0.01 it sees only 1%.
            Masked attack rows are treated as normal in the training target.
        """
        self._feature_cols = [c for c in feat_df.columns if c not in _META_COLS]
        X = feat_df[self._feature_cols].values
        y_true = (feat_df["label"] != "normal").astype(int).values

        if label_ratio < 1.0:
            rng = np.random.default_rng(42)
            attack_idx = np.where(y_true == 1)[0]
            n_keep = max(1, int(len(attack_idx) * label_ratio))
            kept = rng.choice(attack_idx, size=n_keep, replace=False)
            y_train = np.zeros_like(y_true)
            y_train[kept] = 1
            logger.info(
                "Label masking: keeping %d / %d attack labels (label_ratio=%.2f)",
                n_keep, len(attack_idx), label_ratio,
            )
        else:
            y_train = y_true

        n_pos = int(y_train.sum())
        n_neg = int((y_train == 0).sum())
        scale_pos_weight = n_neg / max(n_pos, 1)

        logger.info(
            "Training XGBoost on %d windows (%d normal, %d labeled attack), spw=%.2f …",
            len(X), n_neg, n_pos, scale_pos_weight,
        )

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            verbosity=0,
        )
        self.model.fit(X, y_train)
        self._fitted = True
        logger.info("XGBoost training complete.")

    def predict(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """Run inference.  Returns input DataFrame with pred_flag + anomaly_score."""
        if not self._fitted or self.model is None:
            raise RuntimeError("Model not trained yet. Call .train() first.")

        X = feat_df[self._feature_cols].values
        flags = self.model.predict(X)
        scores = self.model.predict_proba(X)[:, 1]

        out = feat_df.copy()
        out["pred_flag"] = flags.astype(int)
        out["anomaly_score"] = scores
        return out

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("XGBoost model saved → %s", path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> "SupervisedDetector":
        path = Path(path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info("XGBoost model loaded ← %s", path)
        return obj
