# 訓練報告
- 模型: lstm  | 分割: short  | 裝置: cuda  | 參數量: 102018
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 20

## 核心指標
- 最佳 (epoch 20): train_loss=0.2394, train_acc=0.9071, val_loss=0.7709, val_acc=0.7023
- 最終 (epoch 20): train_loss=0.2394, train_acc=0.9071, val_loss=0.7709, val_acc=0.7023
- 一般化落差: at_best=0.2048, at_last=0.2048

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0286
- train_acc_slope: 0.0163
- val_loss_slope: 0.0045
- val_acc_slope: 0.0127

## 過擬合分析
- 判定: 否 (score=1)
- 訊號: early_best=False, loss_rebound=False, gap_large=True, acc_drop=False
- 附註: best_epoch_ratio=1.00

## 設定摘要
- lr: 0.001
- weight_decay: 0.0001
- seed: 42
- use_norm: True
- balance_by_class: True
- amplify_hard_negative: False
- hard_negative_factor: 1.0
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0
