# 訓練報告 (clean no-val 模式)
- 模型: gcn_tcn | 裝置: cuda | 參數量: 94930
- 訓練資料: N=3804 T=30 F=36 | epochs=2 | batch=16
## Epoch 訓練紀錄 (train_loss / train_acc)
- epoch 1: loss=0.6884, acc=0.5405
- epoch 2: loss=0.6701, acc=0.5946

## Test metrics (independent)
- AUC: 0.6611
- F1: 0.6667
- Recall: 0.7313
- Precision: 0.6126
- Composite Score: 0.6978 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6385 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 400
- FP: 253
- FN: 147
- TN: 267

## Top 4 epochs by Composite
- N/A (no validation set used)

## Top 4 epochs by Precision-aware
- N/A (no validation set used)
