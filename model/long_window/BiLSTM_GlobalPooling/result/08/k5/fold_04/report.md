# 訓練報告
- 模型: BiLSTM_GlobalPooling  | 分割: long  | 裝置: cuda  | 參數量: 72321
- 資料: N=1276 T=36 F=75  | 批次: 32  | epoch: 30

## 核心指標
- 最佳 (epoch 4): train_loss=0.6514, val_auc=0.6357, val_f1=0.0364
  - confusion (TP,FP,FN,TN): 2,3,103,147
- 最終 (epoch 4): train_loss=0.6514, val_auc=0.6357, val_f1=0.0364
  - confusion (TP,FP,FN,TN): 2,3,103,147

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0139
- val_auc_slope: 0.0075

## 學習率建議
- 建議: 維持  | 當前 lr: 0.0003
- 理由: 依 val_auc 斜率與 train_loss 變化暫時維持目前 learning rate。

## Top 4 最佳 epoch (以合成分數 0.5*AUC+0.3*F1+0.2*recall 排序)
1. epoch 1: comb_score=0.5044, train_loss=0.6932, val_auc=0.6130, val_f1=0.4311
   - confusion (TP,FP,FN,TN): 36,26,69,124
2. epoch 2: comb_score=0.3562, train_loss=0.6786, val_auc=0.6280, val_f1=0.1026
   - confusion (TP,FP,FN,TN): 6,6,99,144
3. epoch 4: comb_score=0.3326, train_loss=0.6514, val_auc=0.6357, val_f1=0.0364
   - confusion (TP,FP,FN,TN): 2,3,103,147
4. epoch 3: comb_score=0.3321, train_loss=0.6647, val_auc=0.6349, val_f1=0.0364
   - confusion (TP,FP,FN,TN): 2,3,103,147

## 過擬合分析
- 判定: 否 (gap=0.0000)
- 訊號: early_best=True, loss_rebound=False, gap_large=False, acc_drop=False

## 設定摘要
- lr: 0.0003
- weight_decay: 0.01
- seed: 42
- use_bn: False
- pooling: avg
- num_workers: 4