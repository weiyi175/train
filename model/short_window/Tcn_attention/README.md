# Short Window TCN+Attention Baseline

- 目的：以 TCN（dilated causal 1D conv）抽取時序特徵，搭配注意力池化做短視窗二分類（smoke/no_smoke）。
- 輸入：`windows_dense_npz.npz`（short split），建議使用 normalized 特徵。
- 輸出：`result/01`, `02`, ...（自動遞增），含 `config.json`, `train_log.jsonl`, `best.ckpt`, `last.ckpt`, `model_spec.txt`, `report.json`, `report.md`。

## 檔案
- `models.py`：TCN 模組（TemporalBlock/Encoder）+ Attention（Additive/MHSA）+ 分類頭
- `utils.py`：run 目錄遞增、JSON、參數統計、報告產生（含過擬合分析）
- `train_tcn_attn.py`：訓練腳本，載入 `Tool/dataset_npz.py` 的短視窗資料

## 依賴
- PyTorch >= 2.0
- `Tool/dataset_npz.py` 可載入 short split（`windows_dense_npz.npz`）

## 執行
```bash
# 推薦（GPU）：使用 normalized 特徵、類別平衡與強化難負
python /home/user/projects/train/model/short_window/Tcn_attention/train_tcn_attn.py \
  --npz /home/user/projects/train/train_data/slipce/windows_dense_npz.npz \
  --use_norm --epochs 40 --batch_size 256 --lr 1e-3 \
  --balance_by_class --amplify_hard_negative --hard_negative_factor 1.5

# 改用 MHSA（注意力編碼 + 加性池化）
python /home/user/projects/train/model/short_window/Tcn_attention/train_tcn_attn.py \
  --npz /home/user/projects/train/train_data/slipce/windows_dense_npz.npz \
  --use_norm --attn_type mhsa --mhsa_heads 4 --epochs 40
```

## 報告
- 訓練完成後自動產生 `report.json` 與 `report.md`，內含：
  - 最佳/最終指標、一般化落差、趨勢斜率、過擬合判定（score 與訊號）
  - 最佳/最終指標、一般化落差、趨勢斜率、過擬合判定（score 與訊號）