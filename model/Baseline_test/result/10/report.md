# 訓練報告
- 模型: statpool_mlp  | 分割: long  | 裝置: cuda  | 參數量: 85138
- 資料: N=1276 T=75 F=36  | 批次: 256  | epoch: 130

## 核心指標
- 最佳 (epoch 76): train_loss=0.1282, train_acc=0.9569, val_loss=1.2365, val_acc=0.6824
- 最終 (epoch 130): train_loss=0.0569, train_acc=0.9814, val_loss=1.9147, val_acc=0.6392
- 一般化落差: at_best=0.2746, at_last=0.3422

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0016
- train_acc_slope: 0.0003
- val_loss_slope: 0.0139
- val_acc_slope: -0.0013

## 過擬合分析
- 判定: 是 (score=4)
- 訊號: early_best=True, loss_rebound=True, gap_large=True, acc_drop=True
- 附註: best_epoch_ratio=0.58

## 設定摘要
- lr: 0.001
- weight_decay: 0.0001
- seed: 42
- use_norm: True
- concat_raw_norm: False
- balance_by_class: True
- amplify_hard_negative: False
- hard_negative_factor: 1.0
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0
