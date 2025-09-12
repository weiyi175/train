# 訓練報告
- 模型: gcn_tcn  | 分割: short  | 裝置: cuda  | 參數量: 94930
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 100

## 核心指標
- 最佳 (epoch 66): train_loss=0.1995, train_acc=0.9295, val_loss=0.5271, val_acc=0.7653
- 最終 (epoch 100): train_loss=0.1215, train_acc=0.9523, val_loss=0.5688, val_acc=0.7863
- 一般化落差: at_best=-0.3276, at_last=-0.4473

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0028
- train_acc_slope: 0.0014
- val_loss_slope: -0.0005
- val_acc_slope: 0.0019

## 學習率建議
- 建議: 維持  | 當前 lr: 0.0001 
- 理由: val_loss 有下降趨勢，暫時維持目前 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 66: train_loss=0.1995, train_acc=0.9295, val_loss=0.5271, val_acc=0.7653
2. epoch 56: train_loss=0.2636, train_acc=0.8980, val_loss=0.5272, val_acc=0.7595
3. epoch 78: train_loss=0.1865, train_acc=0.9228, val_loss=0.5311, val_acc=0.7767
4. epoch 50: train_loss=0.3016, train_acc=0.8818, val_loss=0.5312, val_acc=0.7424

## 過擬合分析
- 判定: 否 (score=0)
- 訊號: early_best=False, loss_rebound=False, gap_large=False, acc_drop=False
- 附註: best_epoch_ratio=0.66

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