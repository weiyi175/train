# 訓練報告 (clean no-val 模式)
- 模型: gcn_tcn | 裝置: cuda | 參數量: 94930
- 訓練資料: N=3804 T=30 F=36 | epochs=2 | batch=32
## Epoch 訓練紀錄 (train_loss / train_acc)
- epoch 1: loss=0.6886, acc=0.5373
- epoch 2: loss=0.6793, acc=0.5823

## Test metrics (independent)
- AUC: 0.6415
- F1: 0.6204
- Recall: 0.6380
- Precision: 0.6038
- Composite Score: 0.6335 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6163 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 349
- FP: 229
- FN: 198
- TN: 291

## Top 4 epochs by Composite
- N/A (no validation set used)

## Top 4 epochs by Precision-aware
- N/A (no validation set used)
