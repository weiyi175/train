# 訓練報告
- 模型: tcn_attention  | 分割: long  | 裝置: cuda  | 參數量: 991043
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 17
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 17
- NOTE: config.epochs=40 but detected 17 epochs in logs/checkpoints

## 核心指標
- 最佳 (epoch 1): train_loss=0.1066, train_acc=0.9500, val_loss=1.1640, val_acc=0.6706
- 最終 (epoch 17): train_loss=0.0612, train_acc=0.9736, val_loss=1.3486, val_acc=0.6431
- 一般化落差: at_best=-1.0574, at_last=-1.2875

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0000
- train_acc_slope: -0.0004
- val_loss_slope: -0.0027
- val_acc_slope: 0.0022

## 學習率建議
- 建議: 維持  | 當前 lr: 3e-05 
- 理由: val_loss 持續下降，暫時維持目前 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 1: train_loss=0.1066, train_acc=0.9500, val_loss=1.1640, val_acc=0.6706
2. epoch 5: train_loss=0.0888, train_acc=0.9687, val_loss=1.1864, val_acc=0.6745
3. epoch 2: train_loss=0.0804, train_acc=0.9677, val_loss=1.1951, val_acc=0.6627
4. epoch 3: train_loss=0.0775, train_acc=0.9638, val_loss=1.1977, val_acc=0.6471

## 過擬合分析
- 判定: 是 (score=2)
- 訊號: early_best=True, loss_rebound=False, gap_large=False, acc_drop=True
- 附註: best_epoch_ratio=0.06

## 設定摘要
- lr: 3e-05
- weight_decay: 0.0001
- seed: 42
- use_norm: True
- balance_by_class: False
- amplify_hard_negative: False
- hard_negative_factor: 1.5
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0