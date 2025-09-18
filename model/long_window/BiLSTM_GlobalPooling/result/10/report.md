# 訓練報告
- 模型: BiLSTM_GlobalPooling  | 分割: long  | 裝置: cuda  | 參數量: 72321
- 資料: N=1276 T=36 F=75  | 批次: 32  | epoch: 30

## 核心指標
- 最佳 (epoch 4): train_loss=0.6516, val_auc=0.6249
- 最終 (epoch 15): train_loss=0.4048, val_auc=0.5241

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0265
- val_auc_slope: -0.0115

## 學習率建議
- 建議: 維持  | 當前 lr: 0.0003
- 理由: 依 val_auc 斜率與 train_loss 變化暫時維持目前 learning rate。

## Top 4 最佳 epoch (以合成分數 0.5*AUC+0.3*F1+0.2*recall 排序)
1. epoch 7: comb_score=0.4302, train_loss=0.6217, val_auc=0.6177, val_f1=0.4045
   - confusion (TP,FP,FN,TN): 18,19,34,56
2. epoch 10: comb_score=0.4258, train_loss=0.5520, val_auc=0.5544, val_f1=0.4954
   - confusion (TP,FP,FN,TN): 27,30,25,45
3. epoch 11: comb_score=0.4253, train_loss=0.5228, val_auc=0.5505, val_f1=0.5000
   - confusion (TP,FP,FN,TN): 27,29,25,46
4. epoch 8: comb_score=0.4221, train_loss=0.6052, val_auc=0.5964, val_f1=0.4130
   - confusion (TP,FP,FN,TN): 19,21,33,54

## 過擬合分析
- 判定: 是 (gap=0.2468)
- 訊號: early_best=True, loss_rebound=False, gap_large=True, acc_drop=True

## 設定摘要
- lr: 0.0003
- weight_decay: 0.01
- seed: 42
- use_bn: False
- pooling: avg
- num_workers: 4