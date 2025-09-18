## 執行結果報告 (report.md)

### 簡短任務說明
本報告總結最近對 Video Swin (長視窗/StatPool) 與 capacity probe 的開發與執行結果，包含：容量探測 (capacity_probe)、訓練腳本改動、StatPool + Summary + MLP 的實作與目前驗證狀態。

---

### 核心檢查清單
- [x] 實作 capacity_probe 並產出 `capacity_report.json`（已記錄多組 preset + feature_pack 的 best_batch / peak memory / throughput）
- [x] 訓練腳本 `train_videoswin.py` 支援自動採用 capacity_report 中的 batch
- [x] 新增 StatPool pipeline（`dataset_statpool.py`, `model_statpool.py`, `utils_scaler.py`, `train_statpool.py`）並通過語法檢查
- [~] per-sample top-k 高 loss 的精確紀錄（目前為 best-effort；建議在訓練迴圈中改用 loss reduction='none' 以保證精準）

---

### 重要成果摘要
- Capacity probe（代表性結果）:
  - preset=tiny, fp=light/full -> best_batch = 64，peak 約 4.9 GB，throughput 約 270–290 samples/s
  - small/pairwise 模式使記憶體/throughput 上升，pairwise 通常增加噪音

- 訓練實驗摘要（代表）:
  - 使用 capacity_report 自動採 batch（例如 64）後，tiny+light 的若干跑次曾得到驗證 F1 ≈ 0.5204（epoch 1 的 best）
  - full (pairwise) 在部分實驗反而使 F1 降低（噪音增加）

- StatPool pipeline:
  - 已實作 mask-aware StatPool（mean/std/median/max/min）、時間平滑（moving average）、z-score 標準化（per-file / global / none）、以及原骨架衍生的簡易 domain summary（手-口距離 heuristic、mouth_conf）
  - 已支援 WeightedRandomSampler 以平衡類別、也加入 focal loss 選項

---

### 目前狀態（Quality gates）
- Build/語法檢查：PASS（已對 StatPool 檔案執行 compileall，未發現語法錯誤）
- 單元測試：無自動化單元測試（建議新增簡單的 dataset + model smoke tests）
- 訓練驗煙測試：部分訓練已跑完並產出 reproduce.json / best.ckpt / history.json

---

### 變更與產物檔案（重點）
- `model/long_window/StatPool/`:
  - `dataset_statpool.py` — StatPoolFeatureDataset；含 smoothing、z-score、domain summary
  - `model_statpool.py` — MLP 分類器
  - `utils_scaler.py` — JSON 可序列化的 StandardScaler
  - `train_statpool.py` — 訓練腳本（支援 --balance_by_class / --focal_loss / --topk_loss）
  - `report.md`（本檔）
- 其他關聯檔:
  - `scripts/capacity_probe.py` — capacity 探測器，輸出 `capacity_report.json`
  - `scripts/train_videoswin.py` — 支援自動採 batch 與寫出 reproduce artifacts

---

### 如何重現與快速測試
1) 確認虛擬環境並安裝相依（專案已有 venv）：

```bash
source /home/user/projects/train/.venv/bin/activate
```

2) 對 StatPool 做一次小規模 smoke 訓練（範例）：

```bash
python model/long_window/StatPool/train_statpool.py \
  --data_root train_data/Eigenvalue \
  --run_root model/long_window/StatPool/result \
  --epochs 2 --batch 64 --fit_scaler
```

3) 查看產物:
  - `result/0X/history.json`：訓練紀錄
  - `result/0X/best.pt`：最佳模型
  - 若啟用 `--topk_loss`，會產生 top-k loss 的檔案於 run 資料夾

---

### 已知限制與建議（下一步）
1. per-sample 高 loss 精確紀錄：目前採用 best-effort，請在 `train_statpool.py` 的訓練迴圈中把 Loss 的 reduction 設為 `'none'` 並在 batch 結束後計算每個 sample 的 loss，能可靠輸出 top-k；我可以直接修改並提交小 patch。
2. 新增簡單 unit smoke tests：建議新增一個小型 npz / fake dataset 與一個短訓練測試，確保未來改動不會破壞主要流程。
3. domain summary 權重化與 joint index 映射：目前 mouth/hand heuristic 使用約定索引，若要更精準請提供 joint index 映射或想使用的關鍵點命名規則。

---

### 簡短結語
目前所有主要功能已實作並能運行（compile/pass）；若你要我把 per-sample loss 精準化或執行一次完整 StatPool 訓練並把 top-k 結果附上，我會直接修改 `train_statpool.py` 並替你跑一個小型驗證後回傳結果。

報告產生時間：2025-09-13
