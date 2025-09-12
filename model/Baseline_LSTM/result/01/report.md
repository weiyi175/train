# 訓練報告
- 模型: lstm  | 分割: short  | 裝置: cuda  | 參數量: 102018
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 5

## 核心指標
- 最佳 (epoch 4): train_loss=0.6442, train_acc=0.6392, val_loss=0.6952, val_acc=0.5477
- 最終 (epoch 5): train_loss=0.6329, train_acc=0.6468, val_loss=0.6944, val_acc=0.5439
- 一般化落差: at_best=0.0915, at_last=0.1029

## 趨勢 (最後 5 個 epoch 粗略斜率)
- train_loss_slope: -0.0149
- train_acc_slope: 0.0286
- val_loss_slope: 0.0003
- val_acc_slope: 0.0062

## 過擬合分析
- 判定: 否 (score=0)
- 訊號: early_best=False, loss_rebound=False, gap_large=False, acc_drop=False
- 附註: best_epoch_ratio=0.80

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
