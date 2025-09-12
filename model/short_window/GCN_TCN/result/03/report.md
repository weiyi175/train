# 訓練報告
- 模型: gcn_tcn  | 分割: short  | 裝置: cuda  | 參數量: 94930
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 30

## 核心指標
- 最佳 (epoch 29): train_loss=0.1250, train_acc=0.9500, val_loss=0.7000, val_acc=0.7863
- 最終 (epoch 30): train_loss=0.1287, train_acc=0.9471, val_loss=0.7247, val_acc=0.7634
- 一般化落差: at_best=-0.5750, at_last=-0.5960

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0046
- train_acc_slope: 0.0024
- val_loss_slope: -0.0099
- val_acc_slope: 0.0019

## 過擬合分析
- 判定: 否 (score=1)
- 訊號: early_best=False, loss_rebound=False, gap_large=False, acc_drop=True
- 附註: best_epoch_ratio=0.97

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