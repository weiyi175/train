# 訓練報告
- 模型: tcn_attention  | 分割: short  | 裝置: cuda  | 參數量: 473091
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 20
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 20

## 核心指標
- 最佳 (epoch 16): train_loss=0.3178, train_acc=0.8546, val_loss=0.5628, val_acc=0.7347
- 最終 (epoch 20): train_loss=0.2130, train_acc=0.9004, val_loss=0.6469, val_acc=0.7099
- 一般化落差: at_best=-0.2450, at_last=-0.4339

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0300
- train_acc_slope: 0.0154
- val_loss_slope: -0.0071
- val_acc_slope: 0.0083

## 學習率建議
- 建議: 維持  | 當前 lr: 0.0001 
- 理由: val_loss 持續下降，暫時維持目前 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 16: train_loss=0.3178, train_acc=0.8546, val_loss=0.5628, val_acc=0.7347
2. epoch 12: train_loss=0.4391, train_acc=0.7865, val_loss=0.5808, val_acc=0.6966
3. epoch 14: train_loss=0.3891, train_acc=0.8051, val_loss=0.5923, val_acc=0.7118
4. epoch 15: train_loss=0.3243, train_acc=0.8594, val_loss=0.5934, val_acc=0.7042

## 過擬合分析
- 判定: 否 (score=1)
- 訊號: early_best=False, loss_rebound=False, gap_large=False, acc_drop=True
- 附註: best_epoch_ratio=0.80

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