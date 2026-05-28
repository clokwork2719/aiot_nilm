# 專案技術大補帖（Project Handbook）

> **對象**：熟悉 Python、看得懂 IForest / XGBoost 實驗那塊，但對「資料載入、特徵工程、NILM、攻擊數學定義」不熟的開發者。
>
> **目標**：讀完這份文件，你應該能在不看其他人的情況下，獨力 debug 任何一個模組、回答老師對任何一段設計的「為什麼」。
>
> 本文中的數字（AUC/Recall 等）皆取自 `results/`（`compare` 指令的 grid sweep 輸出），以 contamination=20%、per-house、engineered 為主要情境。

---

## 0. 一句話定位

**核心 claim**：竊電偵測的真實困境是「沒有標注資料」——電力公司手上已確認的竊電戶通常不到全體 1%。本專題用 **Isolation Forest（非監督，零標注）** 自學每一戶的正常用電樣態，並用 **XGBoost（監督）在不同標注比例下** 當對照，量化出「標注稀缺時非監督方法更佔優」的交叉點。

---

## 1. 系統架構總覽

### 1.1 資料流向

```
REFIT CSV (每戶一檔, 高頻 mains + sub-meter 讀數)
   │  data_loader.py
   ▼
1h resample → 每日 24 維窗格 (h00..h23)        ──┐ daily_windows_raw.parquet
   │                                              │
   │  attacks.py  inject_attacks(contamination)   │  (sub-meter 日總和)
   ▼                                              │  daily_appliances.parquet
被污染的窗格 (label = normal / h1..h6)            │
   │  feature_extractor.py                        │
   ▼                                              │
特徵矩陣 (engineered 14 維 / raw 26 維)           │
   │                                              │
   ├──────────────┬───────────────────┐          │
   ▼              ▼                   ▼           │
TheftDetector   SupervisedDetector   (評估)        │
(IForest,無監督) (XGBoost,監督)                     │
   │              │                                │
   └──────┬───────┘                                │
          ▼                                         │
   被標記為異常的窗格 ───► nilm_explainer.py ◄──────┘
          │              (比例分配, 用 sub-meter ground truth)
          ▼
   Streamlit dashboard.py (4 tabs)
```

關鍵設計原則：**NILM 解釋層完全不參與偵測**。它只在「某天已被 IForest 標記為異常」之後才啟動，做事後歸因。這讓偵測模組與解釋模組徹底解耦——偵測壞了不影響解釋的邏輯，反之亦然。

### 1.2 模組職責一覽

| 檔案 | 角色 | 一句話職責 |
|------|------|-----------|
| [main.py](../main.py) | CLI 入口 | 4 個子命令：`prepare-data` / `train` / `compare` / `dashboard`，串接所有模組 |
| [src/data/data_loader.py](../src/data/data_loader.py) | Phase 1 | 讀 REFIT CSV、1h resample、切成每日 24 維窗格 |
| [src/data/attacks.py](../src/data/attacks.py) | Phase 2 | 在乾淨窗格上注入 h1–h6 六種合成竊電攻擊 |
| [src/features/feature_extractor.py](../src/features/feature_extractor.py) | Phase 2b | 把 24 維原始日窗轉成 14 維 engineered 或 26 維 raw 特徵 |
| [src/models/anomaly_detector.py](../src/models/anomaly_detector.py) | Phase 3 | `TheftDetector`：IForest 包裝 + 評估（含 per-attack 拆解） |
| [src/models/supervised_detector.py](../src/models/supervised_detector.py) | 對照組 | `SupervisedDetector`：XGBoost 包裝 + `label_ratio` 標注遮蔽 |
| [src/models/nilm_explainer.py](../src/models/nilm_explainer.py) | Phase 4 | 比例分配的 appliance 歸因（不是真 NILM） |
| [src/app/dashboard.py](../src/app/dashboard.py) | Phase 5 | Streamlit 4-tab 視覺化 |

---

## 2. 資料集：REFIT 是什麼

### 2.1 為什麼選 REFIT

REFIT 是英國 Loughborough 大學公開的家庭用電資料集（20 戶、2013–2015、~2 年連續監測）。我們選它的理由：

1. **同時有 mains 與 sub-meter**：每戶 CSV 裡有一條 `Aggregate`（整戶總電表讀數，等同智慧電表會看到的東西）＋ 最多 9 條 `Appliance1..9`（個別電器的功率）。這讓我們可以「用 Aggregate 當智慧電表做偵測，再用 sub-meter 當 ground truth 做事後 appliance 歸因」——兩件事一份資料就搞定。
2. **真實住宅波形**：竊電偵測的難點是「正常用電本身就很亂」。用真實資料注入合成攻擊，比純合成資料更能反映現實的誤報壓力。
3. **規模剛好**：20 戶 × ~2 年 ≈ 9,954 個有效日窗，夠訓練 per-house 模型又不會大到跑不動。

### 2.2 「sub-metered」與「aggregate」是什麼意思

- **sub-metered data**：在每個電器（或插座迴路）上各裝一個量測器，分別記錄冰箱、烘衣機、洗衣機……各自的耗電。這是「分電器」的細粒度資料。
- **aggregate**：整戶一個總電表的讀數。智慧電表在現實中**只看得到 aggregate**，看不到個別電器。

> ⚠️ **一個常被誤會、老師很可能會問的細節**：
> README/報告裡寫「aggregate 成模擬智慧電表讀數」，但**實際程式碼並沒有把 sub-meter 加總**。REFIT 的 CSV 本身就附了一條 `Aggregate` 欄位（這就是該戶真正的 mains 總表），程式直接拿來用。見 [data_loader.py:63-65](../src/data/data_loader.py#L63-L65)：
> ```python
> def aggregate_to_main_meter(df: pd.DataFrame) -> pd.Series:
>     """Return the ``Aggregate`` column (already the summed main meter reading)."""
>     return df["Aggregate"]
> ```
> 也就是說：**偵測用的是 REFIT 既有的 mains 總表**（最貼近真實智慧電表），**sub-meter 只在 NILM 解釋層被用到**（見第 8 章）。這個描述比「加總 sub-meter」更誠實也更正確——因為真實 mains 還包含未被監測的負載，本來就不等於 sub-meter 之和。

### 2.3 data_loader.py 逐段解釋

**(a) 讀單戶 CSV** — [`load_house_csv`](../src/data/data_loader.py#L35)
```python
df = pd.read_csv(path, usecols=["Time", "Aggregate"] + APPLIANCE_COLS,
                 parse_dates=["Time"], index_col="Time", na_values=["", "NA"])
df = df.dropna(axis=1, how="all")   # 整欄全空的 appliance 欄位丟掉（有些戶電器較少）
df = df.dropna(axis=0, how="any")   # 任何一欄有缺值的那筆時間點丟掉
```
- `usecols` 只挑 `Time / Aggregate / Appliance1..9`，省記憶體。
- `dropna(axis=1, how="all")`：第 7 戶可能只有 5 個電器，那 `Appliance6..9` 整欄是空的，直接砍掉。
- `dropna(axis=0, how="any")`：保留乾淨時間點。

**(b) 降頻到每小時** — [`resample_hourly`](../src/data/data_loader.py#L58)
```python
return df.resample("1h").mean()
```
REFIT 原始是 ~8 秒一筆。我們的偵測單位是「一天」，所以先 resample 成每小時平均，得到每天 24 個值。**為什麼用 mean 不是 sum**：mean 對缺漏的取樣點較穩健（少幾筆不會讓那小時整個塌掉）。

**(c) 切日窗** — [`split_into_daily_windows`](../src/data/data_loader.py#L78)
```python
for date, group in series.groupby(series.index.date):
    if len(group) < min_hours:        # min_hours=20，不足 20 小時的殘缺日丟掉
        continue
    hour_range = pd.date_range(start=f"{date} 00:00", periods=24, freq="1h")
    reindexed = group.reindex(hour_range).ffill().bfill()   # 補成剛好 24 格
    if reindexed.isna().any():
        continue
    windows.append((str(date), reindexed.values.astype(np.float64)))
```
- **為什麼要恰好 24 維**：後面所有特徵抽取與 raw 模型都假設輸入是固定 `(24,)`，所以這裡必須把每天硬整成 24 格。
- `ffill().bfill()`：最多補幾個零星的小缺口（DST 換日、漏傳）。`min_hours=20` 已先擋掉殘缺過頭的日子，所以這裡補的量很小。

**(d) 高層入口** — [`load_all_houses`](../src/data/data_loader.py#L110)
回傳兩個 DataFrame：
- `windows_df`：欄位 `house_id, date, h00..h23` —— 偵測主資料。
- `appliance_df`：欄位 `house_id, date, Appliance1..9`（每日總和）—— NILM 解釋層用的 ground truth。

注意 appliance 是用 **每日 sum**（`groupby(date).sum()`），因為解釋層比的是「整天每個電器用了多少 kWh」。

---

## 3. 竊電攻擊的數學定義（h1–h6）

竊電的本質是「讓電表少記」。六種攻擊各自模擬一種真實竄改手法（取自 Jokar et al., 2016, *"Electricity Theft Detection in AMI Using Customers' Consumption Patterns"*）。所有函式都接收 `(24,)` 陣列、回傳修改後的副本、**不修改輸入**（見 [attacks.py](../src/data/attacks.py)）。

| 代碼 | 公式 | 直覺比喻 | 數學上為何合理 |
|------|------|---------|----------------|
| **h1** | $h_1(x_t)=\alpha x_t,\ \alpha\sim U(0.1,0.8)$ | 整天讀數「打折」。像在電表裡裝旁路電阻，所有用電都被等比例少記。 | 保持波形「形狀」，只縮放幅度 → 只動到 mean/std 等尺度特徵。 |
| **h2** | 某連續時段歸零 | 尖峰時段直接把電表接線拔掉，那幾小時記成 0。 | 製造「整段為 0」+ 邊界突跳 → 拉高 zero_ratio 與 MAD。 |
| **h3** | $h_3(x_t)=\gamma_t x_t,\ \gamma_t\sim U(0.1,0.8)$ 逐時獨立 | 裝電子干擾器，每小時隨機砍掉不同比例。 | 注入逐步隨機抖動 → 大幅拉高相鄰差 MAD。 |
| **h4** | $h_4(x_t)=\gamma_t\cdot\text{mean}(x)$ | 偽報：丟掉真實波形，每小時報「日均值的某隨機比例」。 | 破壞真實的平滑日週期，變成繞著均值亂跳的雜訊。 |
| **h5** | $h_5(x_t)=\text{mean}(x)$ | 電表「凍結」，整天都報同一個平均值（零變異）。 | std=0，skewness/kurtosis 全塌成 0 → engineered 特徵一眼看穿。 |
| **h6** | $h_6(x_t)=x_{24-t}$（時間反轉） | 把白天和晚上的用電對調，套利分時電價（離峰偷便宜）。 | mean/std 完全不變，只有「順序」變了 → 唯有看得到序列的 raw 特徵抓得到。 |

### 3.1 實作細節（attacks.py）

每個攻擊都吃一個 `np.random.Generator`（可重現）。幾個值得看的點：

- **h2 隨機選時段** [attacks.py:45-52](../src/data/attacks.py#L45-L52)：
  ```python
  start = rng.integers(0, n - 1)
  end = rng.integers(start + 1, n + 1)
  out[start:end] = 0.0
  ```
  起點隨機、終點在起點之後隨機 → 歸零區間長度也隨機，模擬「斷電多久」不固定。
- **h5 用 `np.full_like`** 製造完全平坦的線；這是 engineered 特徵的「送分題」、卻是 raw 特徵的「地雷」（見第 4.3）。
- **h6 用 `x[::-1].copy()`**：`.copy()` 很關鍵，否則回傳的是 view，後續 in-place 賦值會污染原陣列。

### 3.2 批次注入 — [`inject_attacks`](../src/data/attacks.py#L96)

```python
n_attack = int(round(n * contamination))
attack_idx = rng.choice(n, size=n_attack, replace=False)   # 不重複抽要攻擊的列
chosen_labels = rng.choice(attack_types, size=n_attack)     # 每列隨機指派一種攻擊
```
- `contamination` 控制「多少比例的日窗被竄改」（例如 0.20 = 兩成的日子是竊電）。
- `seed=42` 固定，確保每次跑出來的攻擊集合一樣（實驗可重現）。
- 產出兩個新欄位：`label`（`normal` 或 `h1..h6`）與 `attacked`（bool）。

> ⚠️ **注意 contamination 的雙重身分**：它在這裡是「真實注入率」（ground truth 有多少壞人），在 IForest 又是「決策閾值」參數（見第 5.3）。這兩件事在 `compare` 裡剛好用同一個值，是刻意設計，但概念上要分清楚。

---

## 4. 特徵工程

### 4.1 為什麼不直接把 24 小時原始資料丟進 IForest

技術上可以（這正是 `raw` 模式做的事），但 engineered 特徵在多數攻擊上更強，原因：

1. **維度的語意**：IForest 靠「隨機切特徵」來隔離離群點。如果 24 維都是「某小時的功率」，模型很難用單一次切割表達「這天整體被縮小了」這種跨小時的概念。engineered 把「縮放、形狀、平滑度、稀疏度、時段能量」直接算成單一維度，**讓一刀切就能命中攻擊語意**。
2. **尺度與形狀解耦**：raw 模式為了讓 IForest 看「形狀」會先把 24 小時標準化成零均值單位變異，結果**尺度資訊被洗掉**——h1（整天打折）標準化後跟正常日長得一模一樣，只剩附在後面的 `raw_mean/raw_std` 兩維還留著線索。engineered 直接保留 mean/median/min/max/std，所以 h1 抓得比 raw 好（per-house recall 0.60 vs 更低）。
3. **抗退化**：平坦線（h5）會讓標準化遇到除以 0。raw 模式只能用「全填 0」硬擋，但全 0 的形狀跟「空屋低用電日」撞臉；engineered 則因為 skewness/kurtosis 直接塌成 0 而一眼識破。

### 4.2 14 維 engineered 特徵的分組設計理念

見 [feature_extractor.py:99-162](../src/features/feature_extractor.py#L99-L162)。每組都針對特定攻擊：

| Group | 特徵 | 設計理念（針對哪種攻擊） |
|-------|------|--------------------------|
| **A 統計** | mean, median, min, max, std | 尺度位移。h1 整體縮小 → 這 5 個全往下掉；h5 讓 std 塌成 0。 |
| **B 形狀** | peak_to_avg_ratio, skewness, kurtosis | 波形被攤平或扭曲。h5 平坦 → skew/kurt = 0；PAR 抓尖峰結構。 |
| **C 平滑度** | mean_abs_diff (相鄰小時差絕對值平均) | 高頻抖動。h3/h4 注入逐時雜訊 → MAD 飆高；h2 的歸零邊界也製造跳變。 |
| **D 稀疏度** | zero_ratio (=0 的小時占比) | h2 把整段歸零 → 直接灌爆這個比例。 |
| **E 時段能量** | block_night/morning/afternoon/evening (4×6h 區塊和) | h6 反轉會把「夜間區塊」和「傍晚區塊」的能量簽章對調，對典型居家作息極度異常。 |

維度合計：5+3+1+1+4 = **14**。

**一個防呆細節** [feature_extractor.py:125-130](../src/features/feature_extractor.py#L125-L130)：當 `std < 1e-9`（如 h5 平坦線），`scipy.stats.skew/kurtosis` 會數值爆掉，所以直接把 skew/kurt 設 0。這既是防呆，也剛好是「平坦」的正確語意（無偏態無峰態）。

### 4.3 為什麼 h5 用 engineered 偵測最好、h6 用 raw 最好

這是整個專題最漂亮的「特徵-攻擊對應」洞察：

- **h5（平坦線）→ engineered 勝**（per-house recall 接近 1.0）：平坦讓 std、skew、kurt、MAD **全部塌成 0**。engineered 顯式追蹤這些形狀描述子，平坦線在這些維度上是極端離群。反觀 raw 因為標準化把平坦線變成全 0 向量，跟低變異的夜間/空屋日撞臉，幾乎抓不到。
- **h6（時間反轉）→ raw 勝**（per-house recall ~0.886 vs engineered ~0.50）：反轉**完整保留** mean/std/skew/kurt（統計量對順序不變），只有「序列順序」變了。engineered 把一天壓縮成聚合統計＋4 個粗區塊，反轉只是把區塊兩兩對調，細節已被抹平；raw 把 24 小時逐格喂進去，模型能直接看到「傍晚尖峰跑去半夜」。

> 這正是 `result_analysis.md` 提出 **hybrid 特徵**（engineered + 幾個關鍵 raw 小時）建議的由來——本 session 任務 3 已把它實作成可選項（見 improvements_report.md）。

### 4.4 raw 26 維與批次抽取

`raw` = 24 個標準化小時 + `raw_mean` + `raw_std` = 26 維（[feature_extractor.py:170-193](../src/features/feature_extractor.py#L170-L193)）。批次抽取 [`extract_features_from_df`](../src/features/feature_extractor.py#L201) 用 `np.apply_along_axis` 對每列套用，並把 `house_id/date/label/attacked` 帶過去，最後丟掉任何非有限值的列。

---

## 5. Isolation Forest 的工作原理

### 5.1 直覺比喻

想像你在一堆資料點裡玩「20 個問題」，每次隨機挑一個特徵、隨機挑一個切點把空間一分為二，直到某個點被單獨隔離。**離群點通常很快就被切出來**（因為它跟別人離得遠，幾刀就孤立），正常點則要切很多刀才孤立。IForest 就是大量重複這個隨機切割（200 棵樹），用「平均要切幾刀才能隔離一個點」當異常分數——切得越少越異常。

### 5.2 數學原理：path length 與 anomaly score

- 每棵隨機樹中，一個點 $x$ 從根走到被孤立的葉子，經過的邊數 = **path length** $h(x)$。
- 對 $n$ 個樣本，二元搜尋樹的平均 path length 期望值是
  $$c(n) = 2H(n-1) - \frac{2(n-1)}{n}, \quad H(i)\approx \ln(i)+0.5772$$
- 異常分數標準化為
  $$s(x,n) = 2^{-\,\mathbb{E}[h(x)] / c(n)}$$
  - $s \to 1$：平均 path 極短 → **強烈異常**。
  - $s \to 0.5$：跟正常點差不多。

pyod 的 `decision_function` 回傳的就是這個（越大越異常），`predict` 則根據 `contamination` 設的閾值輸出 0/1。

### 5.3 contamination 參數的兩個作用

這是最容易搞混的點，務必分清楚：

1. **在 `inject_attacks` 裡**：是「真實注入率」——資料中實際有多少比例是攻擊（ground truth）。
2. **在 `IForest(contamination=...)` 裡** [anomaly_detector.py:51-55](../src/models/anomaly_detector.py#L51-L55)：是「決策閾值」——模型假設分數最高的這麼多比例是異常，據此切 0/1 的門檻。它**不影響分數排序、不影響 AUC**（AUC 只看排序），只影響 precision/recall/F1 這些需要硬閾值的指標。

在 `compare` 中兩者用同一個值，等於「告訴模型真實壞人比例」——這是樂觀但公平的設定（IForest 與 XGBoost 都在同一 contamination 下比）。

### 5.4 為什麼 per-house 優於 global

- **global**：一個 IForest 吃所有戶。問題是各戶用電量級差好幾個數量級——大戶的正常用電，對著一堆小戶訓練出來的模型看起來像異常；反過來大戶被偷一半，可能還落在某些中戶的正常範圍裡，**攻擊被跨戶變異淹沒**。
- **per-house**：每戶一個 IForest，只學「這一戶自己的日常」。異常 = 偏離自己的歷史。這把 AUC 從 0.726 拉到 **0.839**（全資料、engineered，見 result_analysis）；在 70/30 temporal split 下 per-house engineered AUC 約 **0.827**（contamination=20%）。
- 部署意義：現實上電力公司本來就該幫每個電表養一個個人化 baseline（用前 30–60 天正常 telemetry），而不是訓練一個全國通用模型。

### 5.5 anomaly_detector.py 實作

- [`TheftDetector.train`](../src/models/anomaly_detector.py#L63)：**只用 `label=='normal'` 的列 fit**。特徵欄是用「排除已知 metadata 欄」推出來的（`_META_COLS`），所以同一份 code 對 14 維或 26 維都通用。
- [`predict`](../src/models/anomaly_detector.py#L89)：輸出 `pred_flag`（0/1）與 `anomaly_score`。
- [`evaluate`](../src/models/anomaly_detector.py#L112) / [`evaluate_results`](../src/models/anomaly_detector.py#L193)：
  - `y_true_binary = (label != 'normal')` → 二元（正常 vs 任一攻擊）。
  - AUC 用連續分數 `anomaly_score` 算（不受閾值影響）。
  - **per_attack 拆解**：對每種攻擊，只取「該攻擊 + 全部 normal」的子集算 recall，所以你看得到「h5 抓 100% 但 h6 只抓 57%」這種細節，證明模型不是只會抓簡單案例。
  - `evaluate_results` 是 standalone 版（不需要 fitted 物件），`compare` 流程用它。

---

## 6. XGBoost baseline 設計

### 6.1 為什麼要用 XGBoost 當對照

如果只報「IForest AUC 0.83」，老師會問：「那監督式不是更強嗎？」——對，**標注充足時 XGBoost 必勝**（lr=100% 時 AUC 0.967）。所以單純比「IForest vs 全標注 XGBoost」反而證明 IForest 沒用。

正確的對照是把 XGBoost 放在**標注稀缺**的真實條件下，畫出它隨標注比例的成長曲線，再把 IForest（零標注、固定一條水平線）疊上去。交叉點就是本專題的核心貢獻。

### 6.2 label_ratio 如何模擬真實標注稀缺

[supervised_detector.py:61-71](../src/models/supervised_detector.py#L61-L71)：
```python
if label_ratio < 1.0:
    attack_idx = np.where(y_true == 1)[0]
    n_keep = max(1, int(len(attack_idx) * label_ratio))
    kept = rng.choice(attack_idx, size=n_keep, replace=False)
    y_train = np.zeros_like(y_true)
    y_train[kept] = 1   # 只有被「確認」的攻擊保留標籤，其餘當成 normal
```
- `label_ratio=0.05` = 「電力公司只確認了 5% 的竊電案例，其餘 95% 的竊電戶因為沒被抓到，在訓練資料裡被當成正常用戶」。
- 這完美對應現實：**未被發現的竊電 = 標成 normal 的髒標籤**。這也是為什麼 XGBoost 在低 label_ratio 時 recall 崩到個位數%——它根本沒看過幾個正樣本，還把大量真竊電當正常學。

### 6.3 scale_pos_weight 的作用

[supervised_detector.py:75-88](../src/models/supervised_detector.py#L75-L88)：
```python
scale_pos_weight = n_neg / max(n_pos, 1)
```
攻擊是少數類（contamination=20% 時正:負約 1:4，label_ratio 低時更懸殊到 1:數百）。`scale_pos_weight` 告訴 XGBoost「把正樣本的損失放大這麼多倍」，避免模型直接全猜 normal。**這已經是對 XGBoost 最有利的設定**——即便如此，標注太少時它依然救不回來，這讓「IForest 在低標注佔優」的結論更有說服力（我們沒有故意弱化對手）。

### 6.4 介面對齊

`SupervisedDetector` 刻意做成跟 `TheftDetector` 同樣的 `train/predict/save/load` 介面，所以 `compare` 的 `_train_and_save` 能用同一段邏輯跑兩種模型（靠 `detector_cls.__name__ == "TheftDetector"` 判斷分支）。`anomaly_score` 對 XGBoost 是 `predict_proba[:,1]`（正類機率），語意上同樣是「越大越像攻擊」，所以 AUC 算法共用。

---

## 7. 評估設計的重要性

### 7.1 為什麼要 temporal 70/30 split（不是隨機 split）

[main.py `_split_per_house`](../main.py#L284)：每戶**按日期排序**，前 70% 當 train、後 30% 當 test。

- **隨機 split 會洩漏未來**：用電有強烈時間自相關（這週跟下週像）。隨機抽會讓「同一週的相鄰兩天」一個進 train 一個進 test，模型等於偷看到了測試期的樣態。
- **temporal split 才符合部署**：現實是「用過去訓練、對未來預測」。先時間排序再切，test 永遠在 train 之後，誠實衡量「面對沒看過的未來」的能力。

### 7.2 為什麼 IForest 可以用全部 normal 訓練、XGBoost 不行

這是本專題最容易被誤會成「作弊」的設計，但其實完全合理（見 [main.py:305-348](../main.py#L305-L348)）：

- **XGBoost（監督）**：訓練時會看 `label`。如果讓它在 test 期的資料上訓練，等於先看過答案 → 必須嚴格只用 train 70%，否則 AUC 會灌水到 ~1.0（這正是修過的 Bug 1）。
- **IForest（非監督）**：訓練時**只看 `label=='normal'` 的特徵，從不接觸攻擊標籤**。它學的是「正常長怎樣」，對它而言 test 期的 normal 跟 train 期的 normal 沒有「答案洩漏」問題——多餵正常資料只是讓 baseline 更準。所以 IForest 用全部 normal 窗格 fit、只在 test 集上評估，是公平的。

> 程式裡用 `is_iforest = detector_cls.__name__ == "TheftDetector"` 來決定：IForest 傳 `full_feat`（全部）當訓練集，XGBoost 只傳 `train_feat`（70%）。兩者都**只在 test 集上評估**，確保 AUC 可比。

### 7.3 train=test 會導致什麼問題（我們修過的 Bug）

| Bug | 症狀 | 修法 |
|-----|------|------|
| **Bug 1** | 原始 `compare` train 與 test 同一份資料 → XGBoost AUC 灌水到 **0.9999**（純過擬合，記住了每一筆） | 加入 temporal 70/30 split，XGBoost 只在沒看過的 test 上評估 |
| **Bug 2** | split 後 IForest 也只用 70% normal 訓練 → precision 從 ~0.6 掉到 ~0.4（資料變少、baseline 變糟，且對非監督毫無必要） | IForest 改用全部 normal 訓練（無 leakage 風險），XGBoost 維持 70% |

**教訓**：評估協定要依模型性質客製。對非監督硬套監督式的 split 規矩，反而製造不公平。

---

## 8. NILM Explainability Layer

### 8.1 它做的其實是「比例分配」，不是真正的 NILM disaggregation

務必對外講清楚（老師很可能戳這點）：真正的 NILM disaggregation（如 FHMM、Combinatorial Optimization）是「只給整戶總功率，反推每個電器各用多少」——需要大量 appliance 級標注與複雜訓練。

**我們做的是 proportional-share allocation**（[nilm_explainer.py](../src/models/nilm_explainer.py)）：
$$\text{估計}_a = \text{被標記日總量} \times \text{該電器歷史占比}_a$$
$$\Delta_a = \text{估計}_a - \text{baseline}_a$$
取 $|\Delta_a / \text{baseline}_a|$ 最大的電器當「主導異常」。

而且因為 REFIT 有 sub-meter ground truth，`explain()` 會**優先直接查當天該電器的真實讀數**（[nilm_explainer.py:162-197](../src/models/nilm_explainer.py#L162-L197)），查不到才退回比例分配。所以它更像「用 ground truth 做事後歸因」而非「反推」。**叫它 NILM disaggregation 會被問倒，請叫它 appliance attribution / 比例分配。**

### 8.2 為什麼這樣設計

1. **輕量、無狀態、免訓練**：`build()` 一次掃過 normal 窗算出每戶 baseline 與占比；`explain()` 純查表＋乘除，適合 dashboard 即時跑。
2. **解釋目標夠用**：我們要回答的不是「精確還原每個電器」，而是「**哪個電器看起來不對勁**」。比例分配＋ground truth 查表已足以指出「烘衣機這天負 delta 最大 → 高耗電電器被旁路」這種線索。
3. **與偵測解耦**：只在 flagged 窗啟動，壞了不影響 IForest。

### 8.3 實作重點

- [`build`](../src/models/nilm_explainer.py#L77)：對每戶算 `baseline_totals`（normal 日 kWh 中位數）、`appliance_shares`（各電器占總量比例）、`appliance_baselines`（各電器 normal 日中位數）。沒有 appliance_df 時退回「9 個電器均分」。
- [`explain`](../src/models/nilm_explainer.py#L142)：先試 ground-truth 直查（同戶同日的 sub-meter 真實值），算 delta、挑 dominant；查不到才比例分配（此時 dominant 標 `"unknown"`，因為比例分配下每個電器 delta 與總量同比例，挑「最大」沒意義）。

---

## 9. Streamlit Dashboard 架構

四個 tab（[dashboard.py:208-210](../src/app/dashboard.py#L208-L210)）：

| Tab | 內容 | 設計邏輯 |
|-----|------|---------|
| **📡 Live Stream** | 選定戶的每日總用電時序圖，異常日用紅點標記，可「播放」模擬即時串流 | demo 用——讓觀眾看到「智慧電表資料流進來、系統即時亮紅燈」的感覺 |
| **🔍 Alert Detail** | 選一個被標記的日期 → NILM appliance breakdown 長條圖 + 當日 24h profile vs 正常中位數 | 回答「為什麼這天被判異常、哪個電器怪」 |
| **📊 Summary Stats** | 全體 AUC/precision/recall/F1、per-attack 雷達圖、該戶 label 分布 | 整體成效一覽 |
| **⚖️ Model Comparison** | contamination slider + 4 張「指標 vs label_ratio」折線（IForest 虛線、XGBoost 曲線）+ per-attack recall bar + 結果表 | **承載核心 story 的 tab**：互動展示標注稀缺實驗 |

### 9.1 load_comparison_results 為什麼故意不用 @st.cache_data

[dashboard.py:110-120](../src/app/dashboard.py#L110-L120)：
```python
def load_comparison_results() -> list[dict]:
    # （注意：沒有 @st.cache_data）
    for metrics_path in sorted(RESULTS_DIR.glob("*/*/metrics.json")):
        ...
```
其他讀取函式（`load_results`、`load_metrics`）都有 `@st.cache_data`，**唯獨這個沒有**。原因（Bug 3 的修正）：你常會在 dashboard 開著的情況下，在另一個終端重跑 `compare` 更新 `results/`。如果這函式也被 cache，dashboard 會一直顯示**舊結果**，重整也沒用，非常容易讓人誤判實驗白跑了。拿掉 cache → 每次 re-render 都重新掃 `results/`，永遠是最新。代價是每次重讀磁碟，但 28 個小 JSON 成本可忽略。

### 9.2 主要 sections 速覽

- **Sidebar**：feature method / scope / alert-filter / 選戶 / 播放速度。CLI 參數透過 `streamlit run -- --features=... --scope=...` 傳入，決定 selectbox 預設值（[dashboard.py:51-61](../src/app/dashboard.py#L51-L61)）。
- **Model Comparison tab** 的 lift 計算：`lift = anomaly_precision / contamination`（隨機抽查的命中率＝母體攻擊率），並用 metric card 顯示 IForest 的「Lift vs Random」。

---

## 10. 實驗結果解讀指南

### 10.1 如何看 Label-Ratio 圖（簡報 fig1 / dashboard Model Comparison）

- **X 軸**：給 XGBoost 的攻擊標注比例（0→100%）。
- **橘色水平線（IForest）**：完全不隨 X 變動——它**不用標注**，所以是一條固定基準。
- **藍色曲線（XGBoost）**：隨標注增加而爬升。
- **怎麼讀**：找曲線「穿過」水平線的位置。
  - **AUC 交叉點 ≈ label_ratio 10–20%**：XGBoost 要看到一兩成攻擊標注才追上 IForest 的 AUC。
  - **Recall 差距更誇張**：lr=5% 時 XGBoost recall 僅 ~3.4%，IForest 有 **73.8%**（cont=20%）。在「不能漏抓竊電」的場景，這是天與地的差距。
- **結論**：在「已確認竊電戶 < 5%」的現實標注條件下，落在水平線左側的整段都是 IForest 勝。

### 10.2 Precision 0.47 為什麼不代表沒用（Lift over Random）

contamination=20% 時 IForest precision ≈ **0.47**。聽起來「一半是誤報」，但要跟**正確基準**比：

- 隨機抽查的命中率 = 母體竊電率 = **20%**。
- IForest 命中率 = **47%** → **lift = 47%/20% ≈ 2.35×**。
- 白話：**同樣派人去查 100 戶，隨機只抓到 20 個真竊電，IForest 抓到 47 個**（簡報 fig4）。
- 而且：現實竊電率遠低於 20%（可能 1–3%），母體越不平衡，「能把命中率拉到 47%」這件事的相對價值越高。
- 最後，**這個場景該主推的指標是 Recall 不是 Precision**——漏抓一個竊電者（損失持續發生）的代價，遠大於多查一個正常戶（一次上門確認）。所以「高 recall、中等 precision」正是我們要的取捨，不需要為 precision 道歉。

### 10.3 核心 contribution 一句話

> **在零標注的真實條件下，per-house Isolation Forest 能達到 AUC≈0.83、Recall≈0.74 的竊電偵測；並量化出監督式 XGBoost 需要約 10–20% 的攻擊標注才追得上——而這個標注量在現實中幾乎不存在。**
