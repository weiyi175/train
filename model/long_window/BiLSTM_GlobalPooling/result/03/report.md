# 訓練報告
- 模型: BiLSTM_GlobalPooling  | 分割: long  | 裝置: cuda  | 參數量: 72321
- 資料: N=1276 T=36 F=75  | 批次: 16  | epoch: 30

## 核心指標
- 最佳 (epoch 4): train_loss=0.6375, val_auc=0.6282
- 最終 (epoch 10): train_loss=0.4674, val_auc=0.5513

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0232
- val_auc_slope: -0.0071

## 學習率建議
- 建議: 維持  | 當前 lr: 0.0003
- 理由: 依 val_auc 斜率與 train_loss 變化暫時維持目前 learning rate。

## Top 4 最佳 epoch (以合成分數 0.5*AUC+0.3*F1+0.2*recall 排序)
1. epoch 7: comb_score=0.4478, train_loss=0.5704, val_auc=0.5874, val_f1=0.5138
   - confusion (TP,FP,FN,TN): 28,29,24,46
2. epoch 8: comb_score=0.4448, train_loss=0.5447, val_auc=0.5954, val_f1=0.4902
   - confusion (TP,FP,FN,TN): 25,25,27,50
3. epoch 6: comb_score=0.4356, train_loss=0.6008, val_auc=0.6003, val_f1=0.4516
   - confusion (TP,FP,FN,TN): 21,20,31,55
4. epoch 10: comb_score=0.4165, train_loss=0.4674, val_auc=0.5513, val_f1=0.4696
   - confusion (TP,FP,FN,TN): 27,36,25,39

## 過擬合分析
- 判定: 是 (gap=0.1701)
- 訊號: early_best=True, loss_rebound=False, gap_large=True, acc_drop=True

## 設定摘要
- lr: 0.0003
- weight_decay: 0.01
- seed: 42
- use_bn: False
- pooling: avg
- num_workers: 4