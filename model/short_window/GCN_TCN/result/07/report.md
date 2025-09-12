# 訓練報告
- 模型: gcn_tcn  | 分割: short  | 裝置: cuda  | 參數量: 94930
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 50

## 核心指標
- 最佳 (epoch 50): train_loss=0.3014, train_acc=0.8813, val_loss=0.5301, val_acc=0.7424
- 最終 (epoch 50): train_loss=0.3014, train_acc=0.8813, val_loss=0.5301, val_acc=0.7424
- 一般化落差: at_best=-0.2288, at_last=-0.2288

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0083
- train_acc_slope: 0.0048
- val_loss_slope: -0.0017
- val_acc_slope: 0.0011

## 學習率建議
- 建議: 維持  | 當前 lr: 0.0001 
- 理由: val_loss 有下降趨勢，暫時維持目前 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 50: train_loss=0.3014, train_acc=0.8813, val_loss=0.5301, val_acc=0.7424
2. epoch 46: train_loss=0.3238, train_acc=0.8789, val_loss=0.5342, val_acc=0.7557
3. epoch 45: train_loss=0.3430, train_acc=0.8589, val_loss=0.5388, val_acc=0.7443
4. epoch 44: train_loss=0.3313, train_acc=0.8627, val_loss=0.5406, val_acc=0.7366

## 過擬合分析
- 判定: 否 (score=0)
- 訊號: early_best=False, loss_rebound=False, gap_large=False, acc_drop=False
- 附註: best_epoch_ratio=1.00

## 設定摘要
- lr: 0.0001
- weight_decay: 0.0001
- seed: 42
- use_norm: True
- balance_by_class: True
- amplify_hard_negative: True
- hard_negative_factor: 1.5
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0