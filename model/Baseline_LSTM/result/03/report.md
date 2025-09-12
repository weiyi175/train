# 訓練報告
- 模型: lstm  | 分割: long  | 裝置: cuda  | 參數量: 102018
- 資料: N=1276 T=75 F=36  | 批次: 256  | epoch: 80

## 核心指標
- 最佳 (epoch 17): train_loss=0.4630, train_acc=0.7747, val_loss=0.7508, val_acc=0.5922
- 最終 (epoch 80): train_loss=0.0200, train_acc=0.9912, val_loss=2.4962, val_acc=0.5686
- 一般化落差: at_best=0.1826, at_last=0.4226

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0021
- train_acc_slope: 0.0001
- val_loss_slope: 0.0082
- val_acc_slope: 0.0017

## 過擬合分析
- 判定: 是 (score=4)
- 訊號: early_best=True, loss_rebound=True, gap_large=True, acc_drop=True
- 附註: best_epoch_ratio=0.21

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
