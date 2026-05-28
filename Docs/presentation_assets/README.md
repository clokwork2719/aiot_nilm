# 簡報用實驗結果圖表

本資料夾的所有 PNG 皆由 [`../generate_presentation_assets.py`](../generate_presentation_assets.py) 從專案根目錄的 `results/`（`cmd_compare` grid sweep 輸出）自動生成。

- 白底、繁體中文、字體放大（title ≥ 16pt、label ≥ 12pt）、200 DPI
- 直接拖入 PowerPoint / Google Slides 即可，不需再處理背景

重新生成指令：

```bash
uv run python docs/generate_presentation_assets.py
```

> 若 `results/` 內容有更新（重跑 compare），重新執行上面指令即可同步所有圖表。

---

## 各圖檔說明與簡報定位

| 檔案 | 內容 | 建議放在簡報的位置 |
|------|------|----------------------|
| `fig1_label_ratio_main.png` | **核心主圖**。雙 Y 軸：XGBoost 的 AUC（藍實線）與 Recall（藍虛線）隨標注比例上升；IForest 為兩條水平參考線（橘色，無需標注）。左側陰影標出「現實可用標注範圍 < 5%」。 | **結果章節的第一張、也是整份簡報的關鍵 slide**。用來講「在現實低標注區間，IForest 完勝 XGBoost」的主故事。建議單獨一頁並口頭強調陰影區。 |
| `fig2_contamination_auc.png` | 四種 contamination（5/10/15/20%）下，IForest、XGBoost(lr=100%)、XGBoost(lr=5%) 的 AUC 折線。 | 緊接在主圖之後，作為**穩健性佐證**：說明「即使換不同竊電率，結論不變——全標注 XGBoost 占優，但現實低標注時 IForest 才是正確選擇」。 |
| `fig3_per_attack_recall.png` | h1~h6 各攻擊類型的 Recall 分組長條圖（IForest / XGBoost lr=5% / lr=100%）。 | **深入分析頁**。展示模型不是只抓簡單案例：IForest 對 h4/h5 近乎完美，h6 較弱；XGBoost 在 lr=5% 時各類型 recall 幾乎全趴。可帶到「特徵工程的強弱項」討論。 |
| `fig4_lift_over_random.png` | 以竊電率 20%、precision 47% 為例，比較「隨機查 100 戶」vs「IForest 查 100 戶」抓到的竊電戶數（20 vs 47，2.35×）。 | 回應老師質疑「precision 0.47 不高怎麼辦」的**防守頁**。用直覺的「同樣查 100 戶」框架說明 lift over random，避免為 precision 道歉。 |
| `table1_summary.png` | 核心指標摘要表（contamination=20%）：IForest 與 XGBoost(lr=1%/5%/100%) 的 AUC/Recall/Precision/F1，IForest 列以橘底突顯。 | **結論頁或 backup**。一張表交代所有關鍵數字，方便 Q&A 時快速指認。 |

---

## 建議的簡報敘事順序

1. （方法頁之後）`fig1` — 拋出核心結論。
2. `fig2` — 證明結論在不同 contamination 下穩健。
3. `fig3` — 拆解到攻擊類型，展示細節與誠實面對弱點（h6）。
4. `fig4` — 預先化解 precision 質疑。
5. `table1` — 收尾與 Q&A 用。

## 關鍵口頭結論（一句話）

> 在電力公司「已確認竊電戶不到 5%」的現實標注條件下，**完全不需要標注資料的 Isolation Forest，其 AUC 與 Recall 都明顯優於必須仰賴標注的 XGBoost**；唯有當標注比例升到約 10–20% 以上，監督式方法才追得上。
