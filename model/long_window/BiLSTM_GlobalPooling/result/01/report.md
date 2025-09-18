# 訓練報告
- 模型: BiLSTM_GlobalPooling  | 分割: long  | 裝置: cuda  | 參數量: 210177
- 資料: N=1276 T=36 F=75  | 批次: 32  | epoch: 30

## 核心指標
- 最佳 (epoch 3): train_loss=0.6484, val_auc=0.6438
- 最終 (epoch 9): train_loss=0.5215, val_auc=0.5846

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0191
- val_auc_slope: -0.0039

## 學習率建議
- 建議: 維持  | 當前 lr: 0.0003
- 理由: 依 val_auc 斜率與 train_loss 變化暫時維持目前 learning rate。

## Top 4 最佳 epoch (以合成分數 0.5*AUC+0.3*F1+0.2*recall 排序)
1. epoch 6: comb_score=0.4963, train_loss=0.5934, val_auc=0.6397, val_f1=0.5882
   - confusion (TP,FP,FN,TN): 35,32,17,43
2. epoch 7: comb_score=0.4865, train_loss=0.5793, val_auc=0.6213, val_f1=0.5862
   - confusion (TP,FP,FN,TN): 34,30,18,45
3. epoch 5: comb_score=0.4771, train_loss=0.6119, val_auc=0.6303, val_f1=0.5400
   - confusion (TP,FP,FN,TN): 27,21,25,54
4. epoch 8: comb_score=0.4539, train_loss=0.5648, val_auc=0.5908, val_f1=0.5283
   - confusion (TP,FP,FN,TN): 28,26,24,49

## 過擬合分析
- 判定: 是 (gap=0.1270)
- 訊號: early_best=True, loss_rebound=False, gap_large=True, acc_drop=True

## 設定摘要
- lr: 0.0003
- weight_decay: 0.01
- seed: 42
- use_bn: False
- pooling: avg
- num_workers: 4