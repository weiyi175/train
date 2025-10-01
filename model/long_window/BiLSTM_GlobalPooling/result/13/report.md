# 訓練報告
- 模型: BiLSTM_GlobalPooling  | 分割: long  | 裝置: cpu  | 參數量: 682497
- 資料: N=1276 T=36 F=75  | 批次: 4  | epoch: 1

## 核心指標
- 最佳 (epoch 1): train_loss=0.6654, val_auc=0.6718
- 最終 (epoch 1): train_loss=0.6654, val_auc=0.6718

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: 0.0000
- val_auc_slope: 0.0000

## 學習率建議
- 建議: 維持  | 當前 lr: 0.001
- 理由: 依 val_auc 斜率與 train_loss 變化暫時維持目前 learning rate。

## Top 4 最佳 epoch (以合成分數 0.5*AUC+0.3*F1+0.2*recall 排序)
1. epoch 1: comb_score=0.4075, train_loss=0.6654, val_auc=0.6718, val_f1=0.2388
   - confusion (TP,FP,FN,TN): 8,7,44,68

## 過擬合分析
- 判定: 否 (gap=0.0000)
- 訊號: early_best=True, loss_rebound=False, gap_large=False, acc_drop=False

## 設定摘要
- lr: 0.001
- weight_decay: 0.01
- seed: 42
- use_bn: False
- pooling: avg
- num_workers: 4

## Independent Test metrics
- AUC: 0.7060
- F1: 0.2130
- Recall: 0.1250
- Precision: 0.7188
- Composite Score: 0.2676 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.5645 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 23
- FP: 9
- FN: 161
- TN: 161

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.6718, f1=0.2388, recall=0.1538, precision=0.5333, composite=0.2829, precisionAware=0.4727  TP:8 FP:7 FN:44 TN:68

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.6718, f1=0.2388, recall=0.1538, precision=0.5333, precisionAware=0.4727, composite=0.2829  TP:8 FP:7 FN:44 TN:68
