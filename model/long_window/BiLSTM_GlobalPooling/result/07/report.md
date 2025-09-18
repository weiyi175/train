# 訓練報告
- 模型: BiLSTM_GlobalPooling  | 分割: long  | 裝置: cuda  | 參數量: 72321
- 資料: N=1276 T=36 F=75  | 批次: 32  | epoch: 30

## 核心指標
- 最佳 (epoch 3): train_loss=0.6612, val_auc=0.5654
- 最終 (epoch 12): train_loss=0.4546, val_auc=0.4985

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0215
- val_auc_slope: -0.0067

## 學習率建議
- 建議: 維持  | 當前 lr: 0.0003
- 理由: 依 val_auc 斜率與 train_loss 變化暫時維持目前 learning rate。

## Top 4 最佳 epoch (以合成分數 0.5*AUC+0.3*F1+0.2*recall 排序)
1. epoch 9: comb_score=0.4178, train_loss=0.5717, val_auc=0.5434, val_f1=0.4870
   - confusion (TP,FP,FN,TN): 28,33,26,40
2. epoch 10: comb_score=0.4015, train_loss=0.5411, val_auc=0.5200, val_f1=0.4717
   - confusion (TP,FP,FN,TN): 25,27,29,46
3. epoch 11: comb_score=0.3913, train_loss=0.5004, val_auc=0.5208, val_f1=0.4364
   - confusion (TP,FP,FN,TN): 24,32,30,41
4. epoch 8: comb_score=0.3867, train_loss=0.5976, val_auc=0.5386, val_f1=0.3913
   - confusion (TP,FP,FN,TN): 18,20,36,53

## 過擬合分析
- 判定: 是 (gap=0.2066)
- 訊號: early_best=True, loss_rebound=False, gap_large=True, acc_drop=True

## 設定摘要
- lr: 0.0003
- weight_decay: 0.01
- seed: 123
- use_bn: False
- pooling: avg
- num_workers: 4