# 訓練報告
- 模型: BiLSTM_GlobalPooling  | 分割: long  | 裝置: cpu  | 參數量: 682497
- 資料: N=1276 T=36 F=75  | 批次: 32  | epoch: 30

## 核心指標
- 最佳 (epoch 1): train_loss=0.6641, val_auc=0.6679
- 最終 (epoch 8): train_loss=0.3506, val_auc=0.5459

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0438
- val_auc_slope: -0.0162

## 學習率建議
- 建議: 維持  | 當前 lr: 0.001
- 理由: 依 val_auc 斜率與 train_loss 變化暫時維持目前 learning rate。

## Top 4 最佳 epoch (以合成分數 0.5*AUC+0.3*F1+0.2*recall 排序)
1. epoch 3: comb_score=0.4855, train_loss=0.6083, val_auc=0.6487, val_f1=0.5370
   - confusion (TP,FP,FN,TN): 29,27,23,48
2. epoch 2: comb_score=0.4802, train_loss=0.6347, val_auc=0.6572, val_f1=0.5053
   - confusion (TP,FP,FN,TN): 24,19,28,56
3. epoch 6: comb_score=0.4762, train_loss=0.4937, val_auc=0.6400, val_f1=0.5208
   - confusion (TP,FP,FN,TN): 25,19,27,56
4. epoch 4: comb_score=0.4670, train_loss=0.5781, val_auc=0.6200, val_f1=0.5234
   - confusion (TP,FP,FN,TN): 28,27,24,48

## 過擬合分析
- 判定: 是 (gap=0.3135)
- 訊號: early_best=True, loss_rebound=False, gap_large=True, acc_drop=True

## 設定摘要
- lr: 0.001
- weight_decay: 0.01
- seed: 42
- use_bn: False
- pooling: avg
- num_workers: 4

## Independent Test metrics
- AUC: 0.6157
- F1: 0.5903
- Recall: 0.5598
- Precision: 0.6242
- Composite Score: 0.5801 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6123 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 103
- FP: 62
- FN: 81
- TN: 108

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 3: auc=0.6487, f1=0.5370, recall=0.5577, precision=0.5179, composite=0.5697, precisionAware=0.5498  TP:29 FP:27 FN:23 TN:48
- epoch 4: auc=0.6200, f1=0.5234, recall=0.5385, precision=0.5091, composite=0.5502, precisionAware=0.5356  TP:28 FP:27 FN:24 TN:48
- epoch 8: auc=0.5459, f1=0.5000, recall=0.5577, precision=0.4531, composite=0.5380, precisionAware=0.4857  TP:29 FP:35 FN:23 TN:40
- epoch 6: auc=0.6400, f1=0.5208, recall=0.4808, precision=0.5682, composite=0.5246, precisionAware=0.5683  TP:25 FP:19 FN:27 TN:56

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 6: auc=0.6400, f1=0.5208, recall=0.4808, precision=0.5682, precisionAware=0.5683, composite=0.5246  TP:25 FP:19 FN:27 TN:56
- epoch 2: auc=0.6572, f1=0.5053, recall=0.4615, precision=0.5581, precisionAware=0.5621, composite=0.5138  TP:24 FP:19 FN:28 TN:56
- epoch 3: auc=0.6487, f1=0.5370, recall=0.5577, precision=0.5179, precisionAware=0.5498, composite=0.5697  TP:29 FP:27 FN:23 TN:48
- epoch 4: auc=0.6200, f1=0.5234, recall=0.5385, precision=0.5091, precisionAware=0.5356, composite=0.5502  TP:28 FP:27 FN:24 TN:48
