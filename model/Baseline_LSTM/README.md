# Baseline_LSTM

LSTM 基線模型，訓練資料使用 `train/Tool/dataset_npz.py` 載入之 dense NPZ（short/long split）。

- 輸出路徑：`model/Baseline_LSTM/result/01`, `02`, ...（自動遞增）
- 自動生成報告：`report.md`、`report.json`

## 使用方式

短視窗（預設 20 epoch、GPU 如可用）

```bash
python model/Baseline_LSTM/train_lstm.py --split short --use_norm --epochs 20 --batch_size 256 --lr 1e-3 --balance_by_class
```

長視窗：

```bash
python model/Baseline_LSTM/train_lstm.py --split long --use_norm --epochs 80 --batch_size 256 --lr 1e-3 --balance_by_class
```

參數：`--hidden`、`--num_layers`、`--bidirectional`、`--dropout`。
