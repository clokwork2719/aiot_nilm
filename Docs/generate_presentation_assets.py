"""
generate_presentation_assets.py
===============================
產生簡報用的實驗結果圖表（白底、繁體中文、>=150 DPI、PNG）。

所有資料皆從專案根目錄的 ``results/`` 讀取（cmd_compare 的 grid sweep 輸出），
不依賴 Streamlit。直接執行即可：

    uv run python docs/generate_presentation_assets.py

輸出位置：docs/presentation_assets/
    fig1_label_ratio_main.png       Label-Ratio 主圖（雙 Y 軸）
    fig2_contamination_auc.png      多 contamination 下的 AUC 比較
    fig3_per_attack_recall.png      Per-Attack Recall 分組長條圖
    fig4_lift_over_random.png       Lift over Random 效益示意圖
    table1_summary.png              核心指標摘要表
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 無顯示環境也能輸出 PNG
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------------------------------------------------------------------------
# 路徑與全域樣式
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / "docs" / "presentation_assets"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DPI = 200  # > 150 DPI，貼簡報清晰

# 顏色（IForest 橘色 / XGBoost 藍色，與任務需求一致）
C_XGB = "#1f6fb2"   # 藍
C_IF = "#e8821e"    # 橘
C_XGB_LO = "#7fb2d6"  # 淺藍（低標注 XGBoost）
C_GREY = "#888888"

# 攻擊類型固定順序
ATTACKS = ["h1", "h2", "h3", "h4", "h5", "h6"]


# ---------------------------------------------------------------------------
# 中文字型設定
# ---------------------------------------------------------------------------

def setup_cjk_font() -> str:
    """註冊系統 CJK 字型並設為 matplotlib 預設，回傳字型名稱。"""
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False  # 負號正常顯示
            print(f"[font] 使用 CJK 字型：{name} ({path})")
            return name
    print("[font] 警告：找不到 CJK 字型，中文可能無法顯示。")
    return "sans-serif"


# ---------------------------------------------------------------------------
# 讀取 results/ metrics
# ---------------------------------------------------------------------------

def load_all_metrics() -> list[dict]:
    """掃描 results/*/*/metrics.json，回傳所有 metrics dict。"""
    records = []
    for p in sorted(RESULTS_DIR.glob("*/*/metrics.json")):
        with open(p) as f:
            records.append(json.load(f))
    if not records:
        raise RuntimeError(
            f"在 {RESULTS_DIR} 找不到任何 metrics.json。"
            "請先執行 `uv run main.py compare ...`。"
        )
    return records


# 簡報圖表固定使用 engineered（專案的 champion 配置）；results/ 內可能同時
# 存在 hybrid 等其他特徵法的結果，因此索引 key 必須含 features 以免互相覆蓋。
FEATURES = "engineered"


def index_metrics(records: list[dict]) -> dict:
    """以 (features, model, contamination, label_ratio) 為 key 建索引。"""
    idx = {}
    for m in records:
        key = (
            m.get("features", "engineered"),
            m["model"],
            round(float(m["contamination"]), 2),
            round(float(m["label_ratio"]), 2),
        )
        idx[key] = m
    return idx


def get(idx: dict, model: str, cont: float, lr: float, features: str = FEATURES) -> dict | None:
    return idx.get((features, model, round(cont, 2), round(lr, 2)))


# ---------------------------------------------------------------------------
# 共用：白底樣式
# ---------------------------------------------------------------------------

def style_axes(ax):
    ax.set_facecolor("white")
    ax.grid(True, color="#dddddd", linewidth=0.8, zorder=0)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=12)


# ---------------------------------------------------------------------------
# 圖 1：Label-Ratio 主圖（contamination=0.20，雙 Y 軸）
# ---------------------------------------------------------------------------

def fig1_label_ratio_main(idx: dict):
    cont = 0.20
    lrs = [0.01, 0.05, 0.10, 0.20, 0.50, 1.0]
    xgb_auc = [get(idx, "xgboost", cont, lr)["auc_roc"] for lr in lrs]
    xgb_rec = [get(idx, "xgboost", cont, lr)["anomaly_recall"] for lr in lrs]
    if_row = get(idx, "iforest", cont, 0.0)
    if_auc = if_row["auc_roc"]
    if_rec = if_row["anomaly_recall"]

    x = [lr * 100 for lr in lrs]  # 百分比

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    style_axes(ax1)
    ax2 = ax1.twinx()

    # XGBoost AUC（藍實線）+ Recall（藍虛線）
    l1, = ax1.plot(x, xgb_auc, "-o", color=C_XGB, linewidth=2.5, markersize=8,
                   label="XGBoost AUC-ROC", zorder=5)
    l2, = ax2.plot(x, xgb_rec, "--s", color=C_XGB, linewidth=2.5, markersize=8,
                   label="XGBoost Recall", zorder=5)

    # IForest 水平參考線
    l3 = ax1.axhline(if_auc, color=C_IF, linewidth=2.5, linestyle="-",
                     label=f"IForest AUC（無需標注資料）= {if_auc:.3f}", zorder=4)
    l4 = ax2.axhline(if_rec, color=C_IF, linewidth=2.5, linestyle="--",
                     label=f"IForest Recall = {if_rec:.3f}", zorder=4)

    # 現實可用標注範圍（< 5%）陰影
    ax1.axvspan(0, 5, color="#f6d3b0", alpha=0.45, zorder=1)
    ax1.annotate(
        "現實可用的標注範圍\n（電力公司確認竊電戶 < 5%）",
        xy=(2.5, 0.30), xycoords=("data", "axes fraction"),
        ha="center", va="center", fontsize=12.5, color="#9a4b00",
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff4e8", ec="#e8821e", lw=1.2),
    )

    ax1.set_xlabel("XGBoost 可用攻擊標注比例 (%)", fontsize=14)
    ax1.set_ylabel("AUC-ROC", fontsize=14, color=C_XGB)
    ax2.set_ylabel("Recall", fontsize=14, color=C_XGB)
    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, 1.0)
    ax1.set_xlim(0, 100)
    ax1.set_title("標注稀缺實驗：IForest vs XGBoost（contamination=20%）",
                  fontsize=17, fontweight="bold", pad=14)

    # 合併圖例
    handles = [l1, l2, l3, l4]
    labels = [h.get_label() for h in handles]
    ax1.legend(handles, labels, loc="upper left", fontsize=11.5, framealpha=0.95)

    fig.tight_layout()
    out = OUT_DIR / "fig1_label_ratio_main.png"
    fig.savefig(out, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig1] 已輸出 → {out}")


# ---------------------------------------------------------------------------
# 圖 2：多 contamination 下的 AUC 比較
# ---------------------------------------------------------------------------

def fig2_contamination_auc(idx: dict):
    conts = [0.05, 0.10, 0.15, 0.20]
    if_auc = [get(idx, "iforest", c, 0.0)["auc_roc"] for c in conts]
    xgb_full = [get(idx, "xgboost", c, 1.0)["auc_roc"] for c in conts]
    xgb_low = [get(idx, "xgboost", c, 0.05)["auc_roc"] for c in conts]

    x = [c * 100 for c in conts]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    style_axes(ax)

    ax.plot(x, if_auc, "-o", color=C_IF, linewidth=2.8, markersize=10,
            label="IForest（0% 標注）", zorder=5)
    ax.plot(x, xgb_full, "-^", color=C_XGB, linewidth=2.8, markersize=10,
            label="XGBoost（lr=100% 全標注）", zorder=5)
    ax.plot(x, xgb_low, "--s", color=C_XGB_LO, linewidth=2.8, markersize=10,
            label="XGBoost（lr=5% 低標注，貼近現實）", zorder=5)

    ax.set_xlabel("Contamination Rate（攻擊注入比例）", fontsize=14)
    ax.set_ylabel("AUC-ROC", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(c)}%" for c in x])
    ax.set_ylim(0.5, 1.0)
    ax.set_title("不同 Contamination 下的 AUC 比較", fontsize=17, fontweight="bold", pad=14)
    ax.legend(loc="lower right", fontsize=12, framealpha=0.95)

    ax.annotate(
        "現實低標注時 XGBoost 崩潰，\nIForest 才是正確選擇",
        xy=(20, get(idx, "xgboost", 0.20, 0.05)["auc_roc"]),
        xytext=(11.5, 0.60), fontsize=12, color="#9a4b00",
        arrowprops=dict(arrowstyle="->", color="#9a4b00", lw=1.6),
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff4e8", ec="#e8821e", lw=1.2),
    )

    fig.tight_layout()
    out = OUT_DIR / "fig2_contamination_auc.png"
    fig.savefig(out, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig2] 已輸出 → {out}")


# ---------------------------------------------------------------------------
# 圖 3：Per-Attack Recall 分組長條圖（contamination=0.20）
# ---------------------------------------------------------------------------

def fig3_per_attack_recall(idx: dict):
    cont = 0.20
    import numpy as np

    def pa_recall(m):
        pa = m.get("per_attack", {})
        return [float(pa.get(a, {}).get("recall", 0.0)) for a in ATTACKS]

    if_rec = pa_recall(get(idx, "iforest", cont, 0.0))
    xgb_lo = pa_recall(get(idx, "xgboost", cont, 0.05))
    xgb_hi = pa_recall(get(idx, "xgboost", cont, 1.0))

    x = np.arange(len(ATTACKS))
    w = 0.26

    fig, ax = plt.subplots(figsize=(11, 6.5))
    style_axes(ax)

    ax.bar(x - w, if_rec, w, label="IForest（0% 標注）", color=C_IF, zorder=3)
    ax.bar(x, xgb_lo, w, label="XGBoost（lr=5%）", color=C_XGB_LO, zorder=3)
    ax.bar(x + w, xgb_hi, w, label="XGBoost（lr=100%）", color=C_XGB, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in ATTACKS], fontsize=13)
    ax.set_xlabel("攻擊類型", fontsize=14)
    ax.set_ylabel("Recall", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.set_title("各攻擊類型 Recall 比較（contamination=20%）",
                 fontsize=17, fontweight="bold", pad=14)
    ax.legend(loc="upper right", fontsize=12, framealpha=0.95)

    fig.tight_layout()
    out = OUT_DIR / "fig3_per_attack_recall.png"
    fig.savefig(out, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig3] 已輸出 → {out}")


# ---------------------------------------------------------------------------
# 圖 4：Lift over Random 效益示意圖
# ---------------------------------------------------------------------------

def fig4_lift_over_random(idx: dict):
    """以 contamination=0.20、IForest precision 為例，
    比較『隨機查 100 戶』vs『IForest 查 100 戶』各能抓到幾個竊電戶。"""
    cont = 0.20
    if_prec = get(idx, "iforest", cont, 0.0)["anomaly_precision"]
    random_prec = cont  # 隨機抽查的命中率 = 母體竊電率
    n_inspect = 100
    random_hits = random_prec * n_inspect
    iforest_hits = if_prec * n_inspect
    lift = if_prec / random_prec

    fig, ax = plt.subplots(figsize=(9, 6.5))
    style_axes(ax)

    bars = ax.bar(
        ["隨機抽查\n100 戶", "IForest 標記\n100 戶"],
        [random_hits, iforest_hits],
        color=[C_GREY, C_IF], width=0.55, zorder=3,
    )
    for b, v in zip(bars, [random_hits, iforest_hits]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2,
                f"{v:.0f} 戶", ha="center", fontsize=15, fontweight="bold")

    ax.set_ylabel("實際抓到的竊電戶數", fontsize=14)
    ax.set_ylim(0, max(iforest_hits, random_hits) * 1.35)
    ax.set_title(
        f"Lift over Random：同樣查 100 戶的效益\n"
        f"（竊電率 {cont:.0%}，IForest precision={if_prec:.0%}）",
        fontsize=16, fontweight="bold", pad=14,
    )
    ax.annotate(
        f"效率提升 {lift:.2f}×",
        xy=(1, iforest_hits), xytext=(0.5, iforest_hits * 0.78),
        fontsize=15, color="#9a4b00", ha="center", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#9a4b00", lw=1.8),
        bbox=dict(boxstyle="round,pad=0.4", fc="#fff4e8", ec="#e8821e", lw=1.4),
    )

    fig.tight_layout()
    out = OUT_DIR / "fig4_lift_over_random.png"
    fig.savefig(out, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[fig4] 已輸出 → {out}")


# ---------------------------------------------------------------------------
# 表格 1：核心指標摘要表（contamination=0.20）
# ---------------------------------------------------------------------------

def table1_summary(idx: dict):
    cont = 0.20
    rows_spec = [
        ("IForest", "0%（無需標注）", get(idx, "iforest", cont, 0.0)),
        ("XGBoost (lr=1%)", "1% 攻擊標注", get(idx, "xgboost", cont, 0.01)),
        ("XGBoost (lr=5%)", "5% 攻擊標注", get(idx, "xgboost", cont, 0.05)),
        ("XGBoost (lr=100%)", "100% 全標注", get(idx, "xgboost", cont, 1.0)),
    ]
    col_labels = ["模型", "標注需求", "AUC-ROC", "Recall", "Precision", "F1"]
    table_data = []
    for name, need, m in rows_spec:
        table_data.append([
            name, need,
            f"{m['auc_roc']:.3f}",
            f"{m['anomaly_recall']:.3f}",
            f"{m['anomaly_precision']:.3f}",
            f"{m['anomaly_f1']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.axis("off")
    ax.set_title("核心指標摘要（contamination = 20%）",
                 fontsize=17, fontweight="bold", pad=18)

    tbl = ax.table(cellText=table_data, colLabels=col_labels,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12.5)
    tbl.scale(1, 2.2)

    n_cols = len(col_labels)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:  # 表頭
            cell.set_facecolor("#1f6fb2")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            name = table_data[r - 1][0]
            if name == "IForest":
                cell.set_facecolor("#fdecd9")  # 橘底突顯 IForest
            else:
                cell.set_facecolor("#f5f8fb" if r % 2 else "#ffffff")

    fig.tight_layout()
    out = OUT_DIR / "table1_summary.png"
    fig.savefig(out, dpi=DPI, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print(f"[table1] 已輸出 → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_cjk_font()
    records = load_all_metrics()
    idx = index_metrics(records)
    print(f"[data] 讀入 {len(records)} 筆 metrics。")

    fig1_label_ratio_main(idx)
    fig2_contamination_auc(idx)
    fig3_per_attack_recall(idx)
    fig4_lift_over_random(idx)
    table1_summary(idx)

    print(f"\n全部完成，輸出於：{OUT_DIR}")


if __name__ == "__main__":
    main()
