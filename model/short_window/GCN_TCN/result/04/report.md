# 訓練報告
- 模型: gcn_tcn  | 分割: short  | 裝置: cuda  | 參數量: 94930
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 30

## 核心指標
- 最佳 (epoch 24): train_loss=0.1209, train_acc=0.9490, val_loss=0.5962, val_acc=0.7901
- 最終 (epoch 30): train_loss=0.1226, train_acc=0.9466, val_loss=0.7059, val_acc=0.7557
- 一般化落差: at_best=-0.4754, at_last=-0.5833

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0050
- train_acc_slope: 0.0022
- val_loss_slope: -0.0120
- val_acc_slope: 0.0015

## 過擬合分析
- 判定: 否 (score=1)
- 訊號: early_best=False, loss_rebound=False, gap_large=False, acc_drop=True
- 附註: best_epoch_ratio=0.80

## 設定摘要
- lr: 0.001
- weight_decay: 0.0001
- seed: 42
- use_norm: True
- balance_by_class: True
- amplify_hard_negative: True
- hard_negative_factor: 1.5
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0