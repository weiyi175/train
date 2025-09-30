# 訓練報告 (clean no-val 模式)
- 模型: gcn_tcn | 裝置: cuda | 參數量: 94930
- 訓練資料: N=3804 T=30 F=36 | epochs=1 | batch=96
## Epoch 訓練紀錄 (train_loss / train_acc)
- epoch 1: loss=0.6902, acc=0.5379

## Test metrics (independent)
- AUC: 0.6031
- F1: 0.1299
- Recall: 0.0731
- Precision: 0.5797
- Composite Score: 0.1961 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.4494 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 40
- FP: 29
- FN: 507
- TN: 491

## Top 4 epochs by Composite
- N/A (no validation set used)

## Top 4 epochs by Precision-aware
- N/A (no validation set used)
