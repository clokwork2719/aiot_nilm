# Session 交接報告

> 給你（組長）醒來後看的。這份 session 在你睡覺時把簡報圖表、技術講義、四項程式改進都做完並驗證過了。以下是重點、檔案清單、要跑的指令、以及還能繼續做的事。

---

## 1. 這個 session 完成了什麼

### 任務 1 — 簡報用實驗結果圖表 ✅
產出 5 張可直接貼進簡報的圖（白底、繁中、200 DPI），全部由一支腳本從 `results/` 自動生成、已視覺檢查中文與版面：
- `fig1_label_ratio_main.png` — **核心主圖**：雙 Y 軸，XGBoost AUC/Recall 隨標注上升，IForest 為水平基準線，左側陰影標出「現實標注 < 5%」。
- `fig2_contamination_auc.png` — 4 種 contamination 下 IForest vs XGBoost(100%) vs XGBoost(5%)。
- `fig3_per_attack_recall.png` — h1~h6 分組長條（IForest / XGBoost lr5% / lr100%）。
- `fig4_lift_over_random.png` — 「查 100 戶」隨機 20 vs IForest 47（2.35×）。
- `table1_summary.png` — 核心指標摘要表。
- 另有 `README.md` 說明每張圖該放簡報哪一頁。

### 任務 2 — 專案技術大補帖 ✅
`docs/project_handbook.md`（~30KB，10 章）：架構流向圖、REFIT、攻擊數學、特徵工程、IForest 原理（含 path-length 公式）、XGBoost、評估設計（含 3 個已修 bug）、NILM、dashboard、結果解讀。**程式碼引用都是可點的連結。**

### 任務 3 — 四項創意改進（都實跑驗證過）✅
1. **Hybrid 特徵**（`--features hybrid`，22 維）：h6 recall **0.571→0.653**，AUC 持平。
2. **`benchmark` 指令**：量測 per-house 模型 ~1.9MB、~23µs/window、~44k windows/sec。
3. **result_analysis.md 補完**：加上 §8 label-scarcity、§9 hybrid、§10 edge footprint。
4. **Dashboard 可調閾值 PR-tradeoff**：Summary Stats tab 新增閾值 slider + PR 曲線 + Lift 卡。

### 任務 4 — 最終審視 ✅
重跑圖表、`AppTest` 跑 dashboard（0 例外）、`benchmark`、`train` 回歸測試全過。

---

## 2. 產出/修改的檔案清單

**新增**
```
Docs/generate_presentation_assets.py        圖表生成腳本
Docs/presentation_assets/                    5 PNG + README.md
Docs/project_handbook.md                     技術大補帖
Docs/improvements_report.md                  改進報告（含驗證輸出）
Docs/session_summary.md                      ← 本檔
results/benchmark.json                        edge benchmark 結果
results/contamination_*/iforest_hybrid_*/     hybrid sweep 結果（28 目錄）
results/contamination_*/xgboost_*_hybrid_*/
```
> 註：macOS 不分大小寫，任務要求的 `docs/` 實際就是既有的 `Docs/`，檔案都在裡面。

**修改**
```
main.py                          +cmd_benchmark、--features 加 hybrid、benchmark 子命令
src/features/feature_extractor.py +extract_hybrid_features (22 維)
src/app/dashboard.py             側邊欄加 hybrid、Summary Stats 加 PR-tradeoff
result_analysis.md               +§8/§9/§10、結論標記 hybrid 已實作
```

**備份**：改動前已 commit `0deda42`，回退用 `git checkout 0deda42 -- <檔案>`。

---

## 3. 你醒來後要跑的指令

```bash
cd "aiot_nilm"

# (1) 看簡報圖（已生成在 Docs/presentation_assets/，要重生才跑這行）
uv run python docs/generate_presentation_assets.py

# (2) 開 dashboard，看新的 PR-tradeoff 與 hybrid 對照
uv run main.py dashboard --features engineered --scope per-house
#   → 📊 Summary Stats tab 最下方：拉動「異常分數閾值」slider
#   → ⚖️ Model Comparison tab：Feature Method 選 hybrid 對照 h6 改善

# (3) 看 edge 部署數據
uv run main.py benchmark --features engineered

# (4) 讀文件
#   Docs/project_handbook.md      — 給不熟的隊友/自己複習
#   Docs/improvements_report.md   — 改了什麼、怎麼驗證的
#   result_analysis.md §8–§10     — 報告要用的 label-scarcity 完整分析
```

> ⚠️ 小提醒：dashboard 的 **tabs 1–3** 若把 Feature Method 切到 `hybrid`，需要先 `uv run main.py prepare-data --features hybrid` + `train --features hybrid --scope per-house`（因為那幾個 tab 讀 `data/hybrid/...`）。**Model Comparison tab 不用**，它直接讀 `results/`，hybrid 已經在裡面了。

---

## 4. 還值得繼續做的事

1. **簡報整合**：把 5 張圖照 `presentation_assets/README.md` 的建議順序排進投影片，主圖 fig1 單獨一頁口頭講陰影區。
2. **Hybrid 的 raw 對照**：目前比了 engineered vs hybrid，可再補 raw 一起進 fig3，三特徵法在 h5/h6 的互補更一目了然（raw 結果未跑，要 `compare --features raw`）。
3. **benchmark 補一張圖**：把模型大小 vs n_estimators、recall 的取捨畫出來，強化「可調輕量化」論點（需小改 benchmark 掃 n_estimators）。
4. **真實低 contamination**：現實竊電率可能 1–3%，可加跑 `--contaminations 0.02 0.03` 看 IForest 在更不平衡時 precision 是否如預期更穩。
5. **口頭防守稿**：handbook §10.2 的 lift-over-random、§7 的「IForest 用全部 normal 無 leakage」是最可能被問的兩點，建議先準備好答法。

---

## 5. 一句話總結這個 session

> 把「最有力的 label-scarcity 故事」完整做成了**可貼的圖、可讀的報告、可跑的程式**；並把報告裡只停在嘴上的 hybrid 建議與 edge 可行性，變成**實測有數字**的東西——全部驗證過、沒有破壞既有 pipeline。
