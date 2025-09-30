# 訓練報告 (clean no-val 模式)
- 模型: gcn_tcn | 裝置: cuda | 參數量: 94930
- 訓練資料: N=3804 T=30 F=36 | epochs=2 | batch=256
## Epoch 訓練紀錄 (train_loss / train_acc)
- epoch 1: loss=0.6932, acc=0.5121
- epoch 2: loss=0.6900, acc=0.5318

## Test metrics (independent)
- AUC: 0.5932
- F1: 0.3059
- Recall: 0.2102
- Precision: 0.5610
- Composite Score: 0.3155 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.4909 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 115
- FP: 90
- FN: 432
- TN: 430

## Top 4 epochs by Composite
- N/A (no validation set used)

## Top 4 epochs by Precision-aware
- N/A (no validation set used)
