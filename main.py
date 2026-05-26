"""
main.py — CLI entry point for the AIoT NILM theft detection pipeline.

Commands
--------
    uv run main.py prepare-data   # Phase 1+2: load, aggregate, inject attacks, extract features
    uv run main.py train          # Phase 3: train IForest + evaluate
    uv run main.py dashboard      # Phase 5: launch Streamlit dashboard

Parameters
----------
    --features {engineered,raw}
        engineered (default): 14-dim hand-crafted features
        raw:                  26-dim normalised hourly + mean/std

    --scope {global,per-house}
        global (default):    one IForest trained on all houses
        per-house:           one IForest per house, trained on that house only

Output layout
-------------
    data/
      daily_windows_raw.parquet       ← shared, Phase 1 output
      {features}/
        daily_windows_attacked.parquet  ← Phase 2 output (scope-independent)
        features.parquet                ← Phase 2b output (scope-independent)
        {scope}/
          results.parquet               ← predictions + hour cols
          metrics.json                  ← evaluation metrics
    models/
      {features}/{scope}/
        iforest.pkl                     ← global model
        house_{id}/iforest.pkl          ← per-house models
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

BASE_DATA_DIR = Path(__file__).parent / "data"
BASE_DATA_DIR.mkdir(exist_ok=True)

WINDOWS_RAW_PATH = BASE_DATA_DIR / "daily_windows_raw.parquet"

HOUR_COLS = [f"h{i:02d}" for i in range(24)]


APPLIANCE_DATA_PATH = BASE_DATA_DIR / "daily_appliances.parquet"


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def _paths(features: str, scope: str = "global") -> dict[str, Path]:
    """Return all output paths for a given (features, scope) combination.

    Feature extraction outputs are shared across scopes (same features.parquet).
    Training outputs (results, metrics, models) are scope-specific.
    """
    feat_dir = BASE_DATA_DIR / features
    feat_dir.mkdir(exist_ok=True)
    scope_dir = feat_dir / scope
    scope_dir.mkdir(exist_ok=True)
    model_dir = Path(__file__).parent / "models" / features / scope
    model_dir.mkdir(parents=True, exist_ok=True)

    return {
        "feat_dir": feat_dir,
        "scope_dir": scope_dir,
        "windows_attacked": feat_dir / "daily_windows_attacked.parquet",
        "features": feat_dir / "features.parquet",
        "results": scope_dir / "results.parquet",
        "metrics": scope_dir / "metrics.json",
        "model_dir": model_dir,
        "model": model_dir / "iforest.pkl",  # used by global scope only
    }


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def cmd_prepare_data(
    house_ids: list[int] | None = None,
    contamination: float = 0.20,
    features: str = "engineered",
    force: bool = False,
) -> None:
    """Phase 1 + 2: Load REFIT, aggregate, inject attacks, extract features.

    This is scope-independent: the same features.parquet is shared by both
    global and per-house training runs.
    """
    from src.data.data_loader import load_all_houses, save_windows
    from src.data.attacks import inject_attacks
    from src.features.feature_extractor import extract_features_from_df

    p = _paths(features)  # scope doesn't affect Phase 1/2 paths
    logging.info("Feature method: %s  →  feat dir: %s", features, p["feat_dir"])

    # Phase 1 — load + aggregate (shared raw windows across all methods/scopes)
    if WINDOWS_RAW_PATH.exists() and APPLIANCE_DATA_PATH.exists() and not force:
        logging.info(
            "Raw windows already exist at %s. Skipping load. (use --force to rerun)",
            WINDOWS_RAW_PATH,
        )
        import pandas as pd
        windows_df = pd.read_parquet(WINDOWS_RAW_PATH)
    else:
        logging.info("=== Phase 1: Loading REFIT dataset ===")
        windows_df, appliance_df = load_all_houses(house_ids=house_ids)
        save_windows(windows_df, WINDOWS_RAW_PATH)
        appliance_df.to_parquet(APPLIANCE_DATA_PATH, index=False)
        logging.info(
            "Saved %d raw daily windows and %d daily appliance records.",
            len(windows_df), len(appliance_df)
        )

    # Phase 2 — attack injection
    logging.info(
        "=== Phase 2: Injecting attacks (contamination=%.0f%%) ===",
        100 * contamination,
    )
    attacked_df = inject_attacks(windows_df, contamination=contamination)
    attacked_df.to_parquet(p["windows_attacked"], index=False)
    logging.info("Saved attacked windows → %s", p["windows_attacked"])

    # Phase 2b — feature extraction
    logging.info("=== Phase 2b: Extracting features [%s] ===", features)
    feat_df = extract_features_from_df(attacked_df, method=features)
    feat_df.to_parquet(p["features"], index=False)
    logging.info(
        "Saved features → %s (%d rows, %d feature cols)",
        p["features"], len(feat_df), feat_df.shape[1],
    )


def cmd_train(
    contamination: float = 0.20,
    features: str = "engineered",
    scope: str = "global",
) -> None:
    """Phase 3: Train IForest, run inference, evaluate."""
    import pandas as pd
    from src.models.anomaly_detector import TheftDetector

    p = _paths(features, scope)

    if not p["features"].exists():
        logging.error(
            "Features not found at %s. Run `prepare-data --features %s` first.",
            p["features"], features,
        )
        sys.exit(1)

    logging.info(
        "=== Phase 3: Training IForest [features=%s, scope=%s] ===", features, scope
    )
    feat_df = pd.read_parquet(p["features"])
    attacked_df = pd.read_parquet(p["windows_attacked"])

    if scope == "global":
        result_feat = _train_global(feat_df, contamination, p)
        metrics = _evaluate_and_save(result_feat, attacked_df, p)

    elif scope == "per-house":
        result_feat = _train_per_house(feat_df, contamination, p)
        metrics = _evaluate_and_save(result_feat, attacked_df, p)

    else:
        logging.error("Unknown scope: %s", scope)
        sys.exit(1)

    logging.info("=== Training complete. ===")


def _train_global(
    feat_df,
    contamination: float,
    p: dict,
):
    """Train one IForest on all normal windows; return result_feat DataFrame."""
    from src.models.anomaly_detector import TheftDetector

    detector = TheftDetector(contamination=contamination)
    detector.train(feat_df)

    logging.info("Running inference on full dataset …")
    result_feat = detector.predict(feat_df)
    detector.save(p["model"])
    return result_feat


def _train_per_house(
    feat_df,
    contamination: float,
    p: dict,
):
    """Train one IForest per house; concatenate results; return result_feat DataFrame."""
    import pandas as pd
    from src.models.anomaly_detector import TheftDetector

    houses = sorted(feat_df["house_id"].unique())
    logging.info("Per-house training across %d houses …", len(houses))

    per_house_results = []
    for house_id in houses:
        mask = feat_df["house_id"] == house_id
        house_feat = feat_df[mask]

        n_normal = (house_feat["label"] == "normal").sum()
        logging.info("  House %d: %d windows (%d normal) …", house_id, len(house_feat), n_normal)

        if n_normal < 10:
            logging.warning("  House %d: too few normal windows, skipping.", house_id)
            continue

        detector = TheftDetector(contamination=contamination)
        detector.train(house_feat)
        result_house = detector.predict(house_feat)

        model_path = p["model_dir"] / f"house_{house_id}" / "iforest.pkl"
        detector.save(model_path)

        per_house_results.append(result_house)

    combined = pd.concat(per_house_results).sort_index()
    logging.info("Per-house training complete. Total rows: %d", len(combined))
    return combined


def _evaluate_and_save(result_feat, attacked_df, p: dict) -> dict:
    """Merge predictions onto attacked windows, save results and metrics."""
    import pandas as pd
    from src.models.anomaly_detector import evaluate_results

    # Build result_df: hour cols + predictions, aligned to result_feat's index
    result_df = attacked_df[["house_id", "date", "label", "attacked"] + HOUR_COLS].copy()
    result_df = result_df.loc[result_feat.index]
    result_df["pred_flag"] = result_feat["pred_flag"].values
    result_df["anomaly_score"] = result_feat["anomaly_score"].values
    result_df.to_parquet(p["results"], index=False)
    logging.info("Saved results → %s", p["results"])

    # Evaluate
    logging.info("=== Evaluation ===")
    feature_cols = [c for c in result_feat.columns
                    if c not in {"house_id", "date", "label", "attacked",
                                 "pred_flag", "anomaly_score"}]
    metrics = evaluate_results(result_feat, feature_cols=feature_cols)

    metrics["confusion_matrix"] = metrics["confusion_matrix"].tolist()
    with open(p["metrics"], "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info("Metrics saved → %s", p["metrics"])
    return metrics


def cmd_dashboard(features: str = "engineered", scope: str = "global", alert_filter: str = "attacks") -> None:
    """Phase 5: Launch Streamlit dashboard."""
    import subprocess
    dashboard_path = Path(__file__).parent / "src" / "app" / "dashboard.py"
    logging.info("Launching Streamlit dashboard [features=%s, scope=%s, alert_filter=%s] …", features, scope, alert_filter)
    subprocess.run(
        [
            "streamlit", "run", str(dashboard_path),
            "--", f"--features={features}", f"--scope={scope}", f"--alert-filter={alert_filter}",
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# Argument parsing helpers
# ---------------------------------------------------------------------------


def _add_features_arg(parser) -> None:
    parser.add_argument(
        "--features",
        choices=["engineered", "raw"],
        default="engineered",
        help=(
            "Feature method: 'engineered' (14-dim, default) or "
            "'raw' (26-dim normalised hourly + mean/std)."
        ),
    )


def _add_scope_arg(parser) -> None:
    parser.add_argument(
        "--scope",
        choices=["global", "per-house"],
        default="global",
        help=(
            "Training scope: 'global' (one model for all houses, default) or "
            "'per-house' (one model trained per house)."
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="AIoT NILM Electricity Theft Detection Pipeline"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # prepare-data
    p_prep = sub.add_parser(
        "prepare-data",
        help="Load REFIT + inject attacks + extract features",
    )
    p_prep.add_argument(
        "--houses", nargs="*", type=int, default=None,
        help="House IDs to process (default: all). Example: --houses 1 2 3",
    )
    p_prep.add_argument(
        "--contamination", type=float, default=0.20,
        help="Fraction of windows to attack (default: 0.20)",
    )
    p_prep.add_argument(
        "--force", action="store_true",
        help="Re-process even if raw windows already exist",
    )
    _add_features_arg(p_prep)

    # train
    p_train = sub.add_parser("train", help="Train IForest + evaluate")
    p_train.add_argument(
        "--contamination", type=float, default=0.20,
        help="IForest contamination parameter (default: 0.20)",
    )
    _add_features_arg(p_train)
    _add_scope_arg(p_train)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Launch Streamlit dashboard")
    _add_features_arg(p_dash)
    _add_scope_arg(p_dash)
    p_dash.add_argument(
        "--alert-filter",
        choices=["all", "attacks"],
        default="attacks",
        help="Alert Details filter: 'all' (all flagged dates) or 'attacks' (only flagged dates that are actual attacks, default).",
    )

    args = parser.parse_args()

    if args.command == "prepare-data":
        cmd_prepare_data(
            house_ids=args.houses,
            contamination=args.contamination,
            features=args.features,
            force=args.force,
        )
    elif args.command == "train":
        cmd_train(
            contamination=args.contamination,
            features=args.features,
            scope=args.scope,
        )
    elif args.command == "dashboard":
        cmd_dashboard(features=args.features, scope=args.scope, alert_filter=args.alert_filter)


if __name__ == "__main__":
    main()
