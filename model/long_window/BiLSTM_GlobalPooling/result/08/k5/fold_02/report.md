# 訓練報告
- 模型: BiLSTM_GlobalPooling  | 分割: long  | 裝置: cuda  | 參數量: 72321
- 資料: N=1276 T=36 F=75  | 批次: 32  | epoch: 30

## 核心指標
- 最佳 (epoch 13): train_loss=0.4175, val_auc=0.6206, val_f1=0.5413
  - confusion (TP,FP,FN,TN): 59,55,45,96
- 最終 (epoch 18): train_loss=0.3015, val_auc=0.5953, val_f1=0.5198
  - confusion (TP,FP,FN,TN): 59,64,45,87

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0282
- val_auc_slope: -0.0013

## 學習率建議
- 建議: 維持  | 當前 lr: 0.0003
- 理由: 依 val_auc 斜率與 train_loss 變化暫時維持目前 learning rate。

## Top 4 最佳 epoch (以合成分數 0.5*AUC+0.3*F1+0.2*recall 排序)
1. epoch 15: comb_score=0.5976, train_loss=0.3725, val_auc=0.6169, val_f1=0.5600
   - confusion (TP,FP,FN,TN): 63,58,41,93
2. epoch 13: comb_score=0.5861, train_loss=0.4175, val_auc=0.6206, val_f1=0.5413
   - confusion (TP,FP,FN,TN): 59,55,45,96
3. epoch 12: comb_score=0.5834, train_loss=0.4506, val_auc=0.6137, val_f1=0.5438
   - confusion (TP,FP,FN,TN): 59,54,45,97
4. epoch 14: comb_score=0.5758, train_loss=0.3815, val_auc=0.6161, val_f1=0.5333
   - confusion (TP,FP,FN,TN): 56,50,48,101

## 過擬合分析
- 判定: 否 (gap=0.1160)
- 訊號: early_best=True, loss_rebound=False, gap_large=True, acc_drop=False

## 設定摘要
- lr: 0.0003
- weight_decay: 0.01
- seed: 42
- use_bn: False
- pooling: avg
- num_workers: 4