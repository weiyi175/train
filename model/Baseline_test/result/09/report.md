# 訓練報告
- 模型: mlp_flat  | 分割: short  | 裝置: cuda  | 參數量: 345202
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 20

## 核心指標
- 最佳 (epoch 18): train_loss=0.0351, train_acc=0.9824, val_loss=2.3360, val_acc=0.5477
- 最終 (epoch 20): train_loss=0.0283, train_acc=0.9881, val_loss=2.5962, val_acc=0.5305
- 一般化落差: at_best=0.4347, at_last=0.4575

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0019
- train_acc_slope: 0.0008
- val_loss_slope: 0.0722
- val_acc_slope: 0.0013

## 過擬合分析
- 判定: 是 (score=2)
- 訊號: early_best=False, loss_rebound=True, gap_large=True, acc_drop=False
- 附註: best_epoch_ratio=0.90

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
