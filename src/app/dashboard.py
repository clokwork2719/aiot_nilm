"""
dashboard.py
============
Streamlit dashboard for AIoT electricity theft detection.

Three views
-----------
1. Live Stream    — scrolling hourly consumption chart with anomaly flags
                    (simulated replay from pre-computed CSV)
2. Alert Detail   — NILM breakdown when a flagged window is selected
3. Summary Stats  — Precision / Recall / F1 per attack type, AUC-ROC

Usage
-----
    uv run streamlit run src/app/dashboard.py
    (or via main.py dashboard sub-command)
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.getcwd())

import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.models.nilm_explainer import NilmExplainer

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AIoT Electricity Theft Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DATA_DIR = Path(__file__).parent.parent.parent / "data"

# Feature method and scope: read from CLI args passed via `streamlit run -- --features=raw --scope=per-house`
_cli_features = "engineered"
_cli_scope = "per-house"
_cli_alert_filter = "attacks"
for _arg in sys.argv:
    if _arg.startswith("--features="):
        _cli_features = _arg.split("=", 1)[1].strip()
    if _arg.startswith("--scope="):
        _cli_scope = _arg.split("=", 1)[1].strip()
    if _arg.startswith("--alert-filter="):
        _cli_alert_filter = _arg.split("=", 1)[1].strip()


# ---------------------------------------------------------------------------
# Helpers / caching
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading results …")
def load_results(features: str, scope: str) -> pd.DataFrame:
    path = BASE_DATA_DIR / features / scope / "results.parquet"
    if not path.exists():
        st.error(
            f"Results not found at `{path}`. "
            f"Run `uv run main.py prepare-data --features {features}` and "
            f"`uv run main.py train --features {features} --scope {scope}` first."
        )
        st.stop()
    df = pd.read_parquet(path)
    df = df.sort_values(["house_id", "date"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner="Loading metrics …")
def load_metrics(features: str, scope: str) -> dict | None:
    path = BASE_DATA_DIR / features / scope / "metrics.json"
    if not path.exists():
        return None
    import json

    with open(path) as f:
        return json.load(f)


@st.cache_resource
def get_explainer(df_all: pd.DataFrame) -> NilmExplainer:
    appliance_path = BASE_DATA_DIR / "daily_appliances.parquet"
    if appliance_path.exists():
        appliance_df = pd.read_parquet(appliance_path)
    else:
        appliance_df = None
    return NilmExplainer.build(df_all, appliance_df)


def make_hour_trace(row: pd.Series) -> list[float]:
    """Extract the 24 hourly readings from a result row."""
    return [row[f"h{i:02d}"] for i in range(24)]


def make_appliance_bar(explanation_dict: dict, label: str) -> go.Figure:
    apps = list(explanation_dict["appliance_flagged"].keys())
    flagged_vals = [explanation_dict["appliance_flagged"][a] for a in apps]
    baseline_vals = [explanation_dict["appliance_baseline"][a] for a in apps]

    fig = go.Figure()
    fig.add_bar(
        name="Baseline (normal)",
        x=apps,
        y=baseline_vals,
        marker_color="#4cc9f0",
    )
    fig.add_bar(
        name=f"Flagged day ({label})",
        x=apps,
        y=flagged_vals,
        marker_color="#f72585",
    )
    fig.update_layout(
        barmode="group",
        title=f"Appliance Breakdown — {explanation_dict.get('date', '')}",
        xaxis_title="Appliance",
        yaxis_title="Estimated Daily Energy (kWh)",
        yaxis=dict(type="log", gridcolor="#1e1e2e"),
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚡ Controls")
    st.markdown("---")

    feature_method = st.selectbox(
        "Feature Method",
        ["engineered", "raw"],
        index=0 if _cli_features == "engineered" else 1,
        help="engineered = 14-dim hand-crafted | raw = 26-dim normalised hourly",
    )

    scope = st.selectbox(
        "Training Scope",
        ["global", "per-house"],
        index=0 if _cli_scope == "global" else 1,
        help="global = one IForest for all houses | per-house = one model per house",
    )

    alert_filter = st.selectbox(
        "Alert Details Filter",
        ["attacks", "all"],
        index=0 if _cli_alert_filter == "attacks" else 1,
        help="attacks = show only flagged dates that are actual attacks | all = show all flagged dates",
    )

    df_all = load_results(feature_method, scope)
    houses = sorted(df_all["house_id"].unique().tolist())
    selected_house = st.selectbox("Select House", houses, index=0)

    replay_speed = st.slider("Replay Speed (days/sec)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    contamination_info = st.empty()
    st.markdown("---")
    st.caption(f"AIoT NILM • features: {feature_method} • scope: {scope} • alert-filter: {alert_filter}")


# ---------------------------------------------------------------------------
# Filter to selected house
# ---------------------------------------------------------------------------

df = df_all[df_all["house_id"] == selected_house].reset_index(drop=True)
n_total = len(df)
n_flagged = int(df["pred_flag"].sum())
contamination_info.metric("Anomaly Rate", f"{100 * n_flagged / max(n_total, 1):.1f}%")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_stream, tab_alert, tab_stats = st.tabs(["📡 Live Stream", "🔍 Alert Detail", "📊 Summary Stats"])


# ── Tab 1: Live Stream ──────────────────────────────────────────────────────
with tab_stream:
    st.subheader(f"House {selected_house} — Simulated Smart Meter Replay")

    col_run, col_reset = st.columns([1, 5])
    run_replay = col_run.button("▶ Run Replay", key="btn_run")
    col_reset.button("⏹ Stop / Reset", key="btn_stop")

    chart_placeholder = st.empty()
    status_placeholder = st.empty()

    # Build static full-series plot first
    dates = df["date"].tolist()
    daily_totals = df[[f"h{i:02d}" for i in range(24)]].sum(axis=1).tolist()
    flags = df["pred_flag"].tolist()
    labels = df["label"].tolist()

    def build_full_chart(up_to: int) -> go.Figure:
        fig = go.Figure()
        x = dates[:up_to]
        y = daily_totals[:up_to]
        f = flags[:up_to]
        lbl = labels[:up_to]

        # Main series
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="Daily Consumption (Wh)",
                line=dict(color="#4cc9f0", width=1.5),
            )
        )

        # Anomaly markers
        ax = [x[i] for i in range(len(x)) if f[i] == 1]
        ay = [y[i] for i in range(len(x)) if f[i] == 1]
        al = [lbl[i] for i in range(len(x)) if f[i] == 1]
        if ax:
            fig.add_trace(
                go.Scatter(
                    x=ax,
                    y=ay,
                    mode="markers",
                    name="Anomaly Flagged",
                    marker=dict(color="#f72585", size=8, symbol="circle"),
                    text=al,
                    hovertemplate="<b>%{x}</b><br>%{y:.0f} Wh<br>Attack: %{text}<extra></extra>",
                )
            )

        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Total Daily Energy (Wh)",
            paper_bgcolor="#0f0f1a",
            plot_bgcolor="#0f0f1a",
            font_color="#e0e0e0",
            xaxis=dict(gridcolor="#1e1e2e"),
            yaxis=dict(gridcolor="#1e1e2e"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        return fig

    if run_replay:
        for i in range(1, n_total + 1):
            if st.session_state.get("btn_stop"):
                break
            chart_placeholder.plotly_chart(build_full_chart(i), width="stretch", key=f"chart_{i}")
            status_placeholder.caption(f"Replaying day {i}/{n_total} …")
            time.sleep(1.0 / replay_speed)
        status_placeholder.caption("✅ Replay complete.")
    else:
        chart_placeholder.plotly_chart(build_full_chart(n_total), width="stretch", key="chart_static")


# ── Tab 2: Alert Detail ─────────────────────────────────────────────────────
with tab_alert:
    st.subheader("Flagged Window — NILM Appliance Breakdown")

    if alert_filter == "attacks":
        flagged_df = df[(df["pred_flag"] == 1) & (df["attacked"] == True)].reset_index(drop=True)
    else:
        flagged_df = df[df["pred_flag"] == 1].reset_index(drop=True)

    if flagged_df.empty:
        if alert_filter == "attacks":
            st.info("No flagged actual attacks for this house.")
        else:
            st.info("No anomalies flagged for this house.")
    else:
        flagged_dates = flagged_df["date"].tolist()
        selected_date = st.selectbox("Select flagged date", flagged_dates, key="alert_date")
        row = flagged_df[flagged_df["date"] == selected_date].iloc[0]

        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("True Label", row["label"])
        col_m2.metric("Anomaly Score", f"{row['anomaly_score']:.4f}")
        col_m3.metric(
            "Daily Total (Wh)",
            f"{sum(row[f'h{i:02d}'] for i in range(24)):.0f}",
        )

        # Build a simple proportional explanation using NilmExplainer
        hour_vals = [row[f"h{i:02d}"] for i in range(24)]
        normal_rows = df[df["label"] == "normal"]

        explainer = get_explainer(df_all)
        explanation = explainer.explain(row)
        expl = explanation._asdict()

        st.plotly_chart(
            make_appliance_bar(expl, row["label"]),
            width="stretch",
        )

        # Highlight the dominant anomaly
        dominant_app = explanation.dominant_anomaly
        if dominant_app != "unknown":
            delta_val = explanation.appliance_delta[dominant_app]
            base_val = explanation.appliance_baseline[dominant_app]
            rel_val = delta_val / (base_val + 1e-9)
            direction = "increase" if delta_val > 0 else "reduction"
            st.warning(
                f"⚠️ **Explainability Insights**: **{dominant_app}** is flagged as the dominant anomaly, "
                f"showing the largest relative deviation from its historical baseline (a {direction} of **{abs(rel_val) * 100:.1f}%** or **{abs(delta_val):.3f} kWh**)."
            )

        # Hourly profile
        fig_hourly = go.Figure()
        fig_hourly.add_trace(
            go.Scatter(
                x=list(range(24)),
                y=hour_vals,
                mode="lines+markers",
                name="Flagged day",
                line=dict(color="#f72585"),
            )
        )
        normal_median = normal_rows[[f"h{i:02d}" for i in range(24)]].median()
        fig_hourly.add_trace(
            go.Scatter(
                x=list(range(24)),
                y=normal_median.tolist(),
                mode="lines",
                name="Normal baseline (median)",
                line=dict(color="#4cc9f0", dash="dash"),
            )
        )
        fig_hourly.update_layout(
            title="Hourly Consumption Profile",
            xaxis_title="Hour of Day",
            yaxis_title="Consumption (W)",
            paper_bgcolor="#0f0f1a",
            plot_bgcolor="#0f0f1a",
            font_color="#e0e0e0",
            xaxis=dict(tickvals=list(range(24)), gridcolor="#1e1e2e"),
            yaxis=dict(gridcolor="#1e1e2e"),
        )
        st.plotly_chart(fig_hourly, width="stretch")


# ── Tab 3: Summary Stats ────────────────────────────────────────────────────
with tab_stats:
    st.subheader("Detection Metrics Across All Houses")

    metrics = load_metrics(feature_method, scope)
    if metrics is None:
        st.info("No metrics file found. Run `uv run main.py train` first.")
    else:
        col_a, col_b, col_c, col_d = st.columns(4)
        overall = metrics.get("overall", {})
        macro = overall.get("macro avg", {})
        col_a.metric("AUC-ROC", f"{metrics.get('auc_roc', 0):.3f}")
        col_b.metric("Precision (macro)", f"{macro.get('precision', 0):.3f}")
        col_c.metric("Recall (macro)", f"{macro.get('recall', 0):.3f}")
        col_d.metric("F1 (macro)", f"{macro.get('f1-score', 0):.3f}")

        st.markdown("### Per-Attack-Type Breakdown")
        per_attack = metrics.get("per_attack", {})
        if per_attack:
            rows = []
            for attack, m in sorted(per_attack.items()):
                rows.append(
                    {
                        "Attack": attack,
                        "Precision": round(m.get("precision", 0), 3),
                        "Recall": round(m.get("recall", 0), 3),
                        "F1": round(m.get("f1-score", 0), 3),
                        "Support": int(m.get("support", 0)),
                    }
                )
            st.dataframe(pd.DataFrame(rows).set_index("Attack"), width="stretch")

            # Radar chart
            attacks = [r["Attack"] for r in rows]
            fig_radar = go.Figure()
            for metric_name, color in [
                ("Precision", "#4cc9f0"),
                ("Recall", "#f72585"),
                ("F1", "#7b2d8b"),
            ]:
                vals = [r[metric_name] for r in rows] + [rows[0][metric_name]]
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=vals,
                        theta=attacks + [attacks[0]],
                        mode="lines+markers",
                        name=metric_name,
                        line_color=color,
                    )
                )
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#0f0f1a",
                    radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2e2e4e"),
                    angularaxis=dict(gridcolor="#2e2e4e"),
                ),
                paper_bgcolor="#0f0f1a",
                font_color="#e0e0e0",
                showlegend=True,
                legend=dict(orientation="h"),
                title="Per-Attack Detection Metrics",
            )
            st.plotly_chart(fig_radar, width="stretch")

    st.markdown("### Attack Type Distribution (This House)")
    label_counts = df["label"].value_counts().reset_index()
    label_counts.columns = ["Label", "Count"]
    fig_dist = px.bar(
        label_counts,
        x="Label",
        y="Count",
        color="Label",
        color_discrete_sequence=px.colors.qualitative.Vivid,
        title="Label Distribution",
    )
    fig_dist.update_layout(
        paper_bgcolor="#0f0f1a",
        plot_bgcolor="#0f0f1a",
        font_color="#e0e0e0",
        showlegend=False,
    )
    st.plotly_chart(fig_dist, width="stretch")
