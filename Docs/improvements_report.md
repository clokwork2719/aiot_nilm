# 改進報告（Improvements Report）

本次 session 在「不破壞現有 pipeline（`train` / `compare` / `dashboard` 維持可用）、不碰系統安全性、每項都實際跑過驗證」的前提下，完成四項改進。改動前已用 git 建立 checkpoint（commit `0deda42`），可隨時回退。

| # | 改進 | 受影響檔案 | 驗證方式 |
|---|------|-----------|----------|
| 1 | Hybrid 特徵法（engineered + 序列錨點） | `src/features/feature_extractor.py`, `main.py`, `src/app/dashboard.py` | `compare --features hybrid` 跑出 metrics，h6 改善 |
| 2 | Edge 部署 benchmark 指令 | `main.py` | `benchmark` 指令實跑，輸出 `results/benchmark.json` |
| 3 | result_analysis.md 補完 label-scarcity 分析 | `result_analysis.md` | 數字對照 `results/` 實際 metrics |
| 4 | Dashboard 可調閾值 PR-tradeoff | `src/app/dashboard.py` | Streamlit `AppTest` 0 例外 + 邏輯獨立測試 |

> 回退指令（如需）：`git checkout 0deda42 -- <檔案>`

---

## 改進 1：Hybrid 特徵法（engineered + 8 個序列錨點）

### 為什麼有價值
`result_analysis.md` 的 §7 早就建議過 hybrid 特徵，但一直停在「建議」。實驗洞察很清楚：engineered 特徵把一天壓成聚合統計，對 **h6（時間反轉）** 幾乎瞎了（recall ~0.57），因為反轉不改變任何統計量、只改變順序；而 raw 特徵雖抓得到 h6，卻因標準化而對 **h5（平坦線）** 失靈。Hybrid 想魚與熊掌兼得：保留 engineered 的尺度/形狀敏感度，再補進少量「序列資訊」讓模型看得到順序。把一個寫在報告裡的 recommendation 變成可跑、可量測的東西，本身就是專題完成度的體現。

### 我做了什麼
- 在 [feature_extractor.py](../src/features/feature_extractor.py) 新增 `extract_hybrid_features()`：14 維 engineered + 每 3 小時取樣的 8 個標準化小時（h00, h03, …, h21）= **22 維**。並加上 `FEATURE_NAMES_HYBRID`、`FEATURE_METHOD_HYBRID`，在 `extract_features_from_df` 加分支。平坦線（std≈0）會走 `np.zeros(24)` 的防呆路徑，與 raw 一致。
- 在 [main.py](../main.py) 的 `_add_features_arg` 把 `--features` 的 choices 從 `[engineered, raw]` 擴成 `[engineered, raw, hybrid]`，`prepare-data` / `train` / `compare` 三個指令同步支援。
- 在 [dashboard.py](../src/app/dashboard.py) 側邊欄 Feature Method selectbox 加入 `hybrid`（Model Comparison tab 本來就動態讀取 `features` 欄，會自動出現 hybrid）。

### 驗證結果
完整 hybrid grid sweep（4 contaminations × 7 runs）跑完 exit 0，產出 28 個 `*hybrid*` 結果目錄。IForest／per-house／temporal split／contamination=20% 下的 per-attack recall：

| 特徵 | AUC | h1 | h2 | h3 | h4 | h5 | h6 |
|------|-----|----|----|----|----|----|----|
| engineered | 0.827 | 0.600 | 0.661 | 0.700 | 0.919 | **1.000** | 0.571 |
| **hybrid** | 0.827 | 0.533 | 0.634 | 0.645 | 0.973 | 0.933 | **0.653** |

**結論（誠實版）**：如假設所預測，**h6 recall 0.571 → 0.653（相對 +14%）**，AUC 持平；代價是 h5 略降（1.0→0.933）與 h1/h3 小幅下滑。所以 hybrid 是個「可調的設計旋鈕」——當分時電價套利（h6）是主要威脅時用 hybrid，當電表凍結（h5）是主要威脅時用 engineered。**不是免費午餐，據實呈現。**

### 使用者如何看到效果
```bash
# 跑 hybrid（已跑過，結果已在 results/）
uv run main.py compare --features hybrid --contaminations 0.20 --label-ratios 0.05 1.0 --scope per-house

# 在 dashboard 的 Model Comparison tab，Feature Method 選 hybrid 即可對照
uv run main.py dashboard --features engineered --scope per-house
```

---

## 改進 2：Edge 部署 benchmark 指令

### 為什麼有價值
`result_analysis.md` §7 宣稱「應部署輕量的 per-meter IForest」，但沒有任何數字佐證「輕量」。老師很可能問：「這真的能跑在邊緣裝置上嗎？多大？多快？」一個量測指令把口號變成可防守的數據。注意：專題主敘事仍是**雲端**，這個 benchmark 只是補上「模型也輕到能放邊緣」的佐證，不是要做 Raspberry Pi demo。

### 我做了什麼
在 [main.py](../main.py) 新增 `cmd_benchmark()` 與 `benchmark` 子命令。它對每一戶訓練一個 IForest，量測：
- **模型序列化大小**（`pickle.dumps` 的 byte 數）：mean / max / 全戶總和。
- **單窗推論延遲**（µs/window）：對每戶重複 `--repeats` 次取 best-of 降噪。
- **吞吐量**（windows/sec）。
結果印成表格並存到 `results/benchmark.json`。完全獨立，不影響其他指令。

### 驗證結果
```
========================================================
  Edge-Deployment Benchmark — features=engineered
========================================================
  Per-house IForest models trained : 20
  Feature dimensionality           : 14
  Model size  (mean / max / total) : 1913.7 / 2275.4 / 38273.4 KB
  Inference latency  (mean / max)  : 22.71 / 27.46 µs/window
  Throughput (mean)                : 44035 windows/sec
========================================================
```
**讀法**：智慧電表一天只產生 1 個窗格，模型 23µs 就分類完，延遲完全不是問題。1.9MB/戶 來自 `n_estimators=200`；若邊緣儲存吃緊，降到 50–100 棵樹可大幅縮小且幾乎不掉準。

### 使用者如何看到效果
```bash
uv run main.py benchmark --features engineered
# 或量測 hybrid / 不同 contamination：
uv run main.py benchmark --features hybrid --contamination 0.10 --repeats 7
```

---

## 改進 3：result_analysis.md 補完 label-scarcity 分析

### 為什麼有價值
原本的 `result_analysis.md` 只有 full-data（train=test，AUC 0.839）的特徵分析，**完全沒寫到本專題最有力的故事——label-scarcity**。記憶與隊友共識都指出這是核心 contribution，報告卻沒收錄，等於把最強的牌藏起來。

### 我做了什麼
在 [result_analysis.md](../result_analysis.md) 補上：
- 一段 **evaluation protocol 說明**，講清楚 §3 的 0.839 是 full-data（樂觀），§8+ 的 0.827 是誠實的 temporal 70/30 split，並解釋為何 IForest 可用全部 normal 訓練而無 leakage。
- **§8 Label-Scarcity Experiment**：完整 7 列 XGBoost vs IForest 表格（取自 `results/` 實際數字）、AUC 交叉點（10–20%）、recall 差距（5% 標注時 3.4% vs 73.8%）、precision 0.47 的 lift-over-random 解讀。
- **§9 Hybrid**（改進 1 的結果表）與 **§10 Edge Footprint**（改進 2 的數據），並把 §7 結論第 2 點標記為「已實作」。

所有數字都與 `results/` 對齊，可重現指令一併附上。

### 驗證結果
表格數字逐一對照 `results/contamination_0.20/*/metrics.json`：IForest AUC 0.827 / recall 0.738、XGBoost lr=1% recall 0.010、lr=5% recall 0.034、lr=100% AUC 0.967 …… 全部一致（見任務 1 抽出的 metrics 摘要）。

### 使用者如何看到效果
直接閱讀 [result_analysis.md](../result_analysis.md) §8–§10。

---

## 改進 4：Dashboard 可調閾值的 Precision–Recall Trade-off

### 為什麼有價值
IForest 的 `contamination` 只決定了**一個**固定切點，但實務上「要多積極標記」是可調的營運決策。一個可拉動閾值、即時看 precision/recall/F1 與 PR 曲線的互動元件，能在 demo 時直接回應老師「precision 不高怎麼辦」——把它變成「我們可以滑到高 recall 端，因為漏抓的代價更高」的現場示範。（Model Comparison tab 已有 Lift vs Random 卡片，故本項聚焦在「可調閾值 + PR 曲線」這個新角度。）

### 我做了什麼
在 [dashboard.py](../src/app/dashboard.py) 的 **📊 Summary Stats** tab 加入「Precision–Recall Trade-off（可調閾值）」區塊：
- 一個異常分數閾值 slider（預設＝重現目前 `pred_flag` 的切點）。
- 即時 precision / recall / F1 / Lift vs Random 四張 metric 卡。
- 一條 PR 曲線（`sklearn.precision_recall_curve`），標出目前操作點與隨機基準水平線。
- 邊界防呆：若該設定下沒有同時包含正常與攻擊樣本，顯示 info 而非崩潰。

### 驗證結果
無法手動點擊 Streamlit，改用**官方 `AppTest` 框架** headless 跑完整 script：
```
AppTest exceptions: 0
tabs: 4
sliders: ['異常分數閾值（≥ 視為竊電）', 'Replay Speed (days/sec)']
selectboxes: [..., 'Feature Method', ...]
```
另外把 PR 計算邏輯抽出獨立跑 `data/engineered/per-house/results.parquet`：`prec 0.481 / rec 0.741 / f1 0.583 / PR 曲線 9820 點`，與已知 full-data 結果一致。`cmd_train` 也回歸測試過（per-house 20 戶訓練完成、無 error）。

### 使用者如何看到效果
```bash
uv run main.py dashboard --features engineered --scope per-house
# 開瀏覽器 → 📊 Summary Stats tab → 最下方「Precision–Recall Trade-off」，拉動閾值
```

---

## 驗證總結

| 項目 | 狀態 |
|------|------|
| `train` 仍可執行 | ✅ 回歸測試通過（per-house 20 戶） |
| `compare` 仍可執行 | ✅ 完整 hybrid sweep exit 0 |
| `dashboard` 仍可執行 | ✅ AppTest 0 例外、4 tabs |
| `benchmark`（新） | ✅ 實跑、輸出 JSON |
| 三種特徵法皆可抽取 | ✅ shape 驗證 (14 / 26 / 22) |
| 簡報圖表不受 hybrid 污染 | ✅ 圖表腳本已用 features 為 key 鎖定 engineered |
| 無系統安全性改動 | ✅ 僅資料分析與視覺化 |
