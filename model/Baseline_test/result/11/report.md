# 訓練報告
- 模型: mlp_flat  | 分割: short  | 裝置: cuda  | 參數量: 345202
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 13

## 核心指標
- 最佳 (epoch 5): train_loss=0.2817, train_acc=0.8856, val_loss=1.0360, val_acc=0.5477
- 最終 (epoch 13): train_loss=0.0444, train_acc=0.9795, val_loss=2.2120, val_acc=0.5019
- 一般化落差: at_best=0.3379, at_last=0.4776

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0372
- train_acc_slope: 0.0157
- val_loss_slope: 0.1388
- val_acc_slope: -0.0013

## 過擬合分析
- 判定: 是 (score=4)
- 訊號: early_best=True, loss_rebound=True, gap_large=True, acc_drop=True
- 附註: best_epoch_ratio=0.38

## 設定摘要
- lr: 0.001
- weight_decay: 0.0001
- seed: 42
- use_norm: True
- concat_raw_norm: False
- balance_by_class: True
- amplify_hard_negative: True
- hard_negative_factor: 1.5
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0
