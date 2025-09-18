# 訓練報告
- 模型: BiLSTM_GlobalPooling  | 分割: long  | 裝置: cuda  | 參數量: 72321
- 資料: N=1276 T=36 F=75  | 批次: 32  | epoch: 30

## 核心指標
- 最佳 (epoch 2): train_loss=0.6768, val_auc=0.5543
- 最終 (epoch 4): train_loss=0.6475, val_auc=0.5404

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0151
- val_auc_slope: 0.0020

## 學習率建議
- 建議: 維持  | 當前 lr: 0.0003
- 理由: 依 val_auc 斜率與 train_loss 變化暫時維持目前 learning rate。

## Top 4 最佳 epoch (以合成分數 0.5*AUC+0.3*F1+0.2*recall 排序)
1. epoch 1: comb_score=0.3697, train_loss=0.6926, val_auc=0.5321, val_f1=0.3457
   - confusion (TP,FP,FN,TN): 14,12,41,60
2. epoch 2: comb_score=0.2873, train_loss=0.6768, val_auc=0.5543, val_f1=0.0339
   - confusion (TP,FP,FN,TN): 1,3,54,69
3. epoch 3: comb_score=0.2853, train_loss=0.6610, val_auc=0.5495, val_f1=0.0351
   - confusion (TP,FP,FN,TN): 1,1,54,71
4. epoch 4: comb_score=0.2805, train_loss=0.6475, val_auc=0.5404, val_f1=0.0345
   - confusion (TP,FP,FN,TN): 1,2,54,70

## 過擬合分析
- 判定: 否 (gap=0.0293)
- 訊號: early_best=True, loss_rebound=False, gap_large=False, acc_drop=False

## 設定摘要
- lr: 0.0003
- weight_decay: 0.01
- seed: 0
- use_bn: False
- pooling: avg
- num_workers: 4