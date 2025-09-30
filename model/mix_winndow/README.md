# 混合視窗離線融合 (short + long)

本模組實作「離線融合」：先各自訓練短視窗 (GCN_TCN) 與長視窗 (CNN_BiLSTM / 其它) 模型，
再將它們對相同樣本輸出的機率或 logits 匯出，使用簡單可學融合層提升最終分類效果。

## 流程概述
1. 個別模型訓練：
   - 短視窗: `model/short_window/GCN_TCN/train_gcn_tcn.py`
   - 長視窗: `model/long_window/CNN_BiLSTM/train_cnn_bilstm.py` (或其他)
2. 匯出對齊資料 (同一順序 N 筆)：
   - `short_probs.npy` (float32, shape (N,)) 或 `short_logits.npy`
   - `long_probs.npy`  (float32, shape (N,)) 或 `long_logits.npy`
   - `labels.npy`      (int64,  shape (N,))
3. 使用本模組 `train_mix.py` 訓練融合層：
   - 支援模式：
     - `avg`：單純平均 (無訓練)
     - `weighted`：學習 2 個權重 (softmax 正規化)
     - `mlp`：MLP 輸入擴增特徵 [p_s, p_l, p_s*p_l, |p_s-p_l|]
     - `stack_logistic`：邏輯迴歸 (torch BCE)
4. (可選) 溫度縮放 + 閾值掃描 (0.05~0.95) 產出最佳 F1 / Composite。

## 主要檔案
- `mix_dataset.py`：載入 probs / logits 與 labels，提供 Dataset。
- `models_mixed.py`：融合模型類別。
- `train_mix.py`：訓練與評估腳本。

## Quick Start
```bash
python train_mix.py \
  --short-probs path/to/short_probs.npy \
  --long-probs path/to/long_probs.npy \
  --labels path/to/labels.npy \
  --fusion mlp --epochs 50 --lr 1e-3 --val-ratio 0.2
```

## 閾值掃描
執行時加上 `--sweep-thresholds` 會輸出 `threshold_metrics.csv` 與最佳摘要。

## 後續擴充 (未實作)
- 直接載入原始 npz 並在一個 DataLoader 中同時取出短/長特徵再分別前向 (需要長視窗模型 Torch 版)
- 使用校準 (溫度縮放) 生成更平滑的機率
- 支援多模型 (>=2) 泛化至任意數量輸入

