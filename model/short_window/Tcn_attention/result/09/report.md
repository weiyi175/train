# 訓練報告 (clean no-val 模式)
- 模型: tcn_attention | 裝置: cuda | 參數量: 473091
- 訓練資料: N=3804 T=30 F=36 | epochs=5 | batch=256
## Epoch 訓練紀錄 (train_loss / train_acc)
- epoch 1: loss=0.6951, acc=0.5208
- epoch 2: loss=0.6877, acc=0.5371
- epoch 3: loss=0.6801, acc=0.5618
- epoch 4: loss=0.6683, acc=0.5936
- epoch 5: loss=0.6463, acc=0.6233

## Test metrics (independent)
- AUC: 0.6610
- F1: 0.4666
- Recall: 0.3638
- Precision: 0.6503
- Composite Score: 0.4541 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.5973 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 199
- FP: 107
- FN: 348
- TN: 413

## Top 4 epochs by Composite
- N/A (no validation set used)

## Top 4 epochs by Precision-aware
- N/A (no validation set used)
