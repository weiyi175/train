# 訓練報告
- 模型: BiLSTM_GlobalPooling  | 分割: long  | 裝置: cuda  | 參數量: 72321
- 資料: N=1276 T=36 F=75  | 批次: 32  | epoch: 30

## 核心指標
- 最佳 (epoch 4): train_loss=0.6537, val_auc=0.5980, val_f1=0.0545
  - confusion (TP,FP,FN,TN): 3,2,102,149
- 最終 (epoch 4): train_loss=0.6537, val_auc=0.5980, val_f1=0.0545
  - confusion (TP,FP,FN,TN): 3,2,102,149

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0129
- val_auc_slope: 0.0075

## 學習率建議
- 建議: 維持  | 當前 lr: 0.0003
- 理由: 依 val_auc 斜率與 train_loss 變化暫時維持目前 learning rate。

## Top 4 最佳 epoch (以合成分數 0.5*AUC+0.3*F1+0.2*recall 排序)
1. epoch 1: comb_score=0.3779, train_loss=0.6927, val_auc=0.5755, val_f1=0.2055
   - confusion (TP,FP,FN,TN): 15,26,90,125
2. epoch 4: comb_score=0.3211, train_loss=0.6537, val_auc=0.5980, val_f1=0.0545
   - confusion (TP,FP,FN,TN): 3,2,102,149
3. epoch 2: comb_score=0.3146, train_loss=0.6781, val_auc=0.5847, val_f1=0.0550
   - confusion (TP,FP,FN,TN): 3,1,102,150
4. epoch 3: comb_score=0.3038, train_loss=0.6656, val_auc=0.5924, val_f1=0.0189
   - confusion (TP,FP,FN,TN): 1,0,104,151

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