# 訓練報告 (clean no-val 模式)
- 模型: gcn_tcn | 裝置: cuda | 參數量: 94930
- 訓練資料: N=3804 T=30 F=36 | epochs=2 | batch=128
## Epoch 訓練紀錄 (train_loss / train_acc)
- epoch 1: loss=0.6913, acc=0.5208
- epoch 2: loss=0.6875, acc=0.5526

## Test metrics (independent)
- AUC: 0.6143
- F1: 0.6488
- Recall: 0.7093
- Precision: 0.5978
- Composite Score: 0.6722 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6164 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 388
- FP: 261
- FN: 159
- TN: 259

## Top 4 epochs by Composite
- N/A (no validation set used)

## Top 4 epochs by Precision-aware
- N/A (no validation set used)
