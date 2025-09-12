# 訓練報告
- 模型: tcn_attention  | 分割: short  | 裝置: cuda  | 參數量: 473091
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 100
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 100

## 核心指標
- 最佳 (epoch 69): train_loss=0.0270, train_acc=0.9857, val_loss=1.0675, val_acc=0.7882
- 最終 (epoch 100): train_loss=0.0328, train_acc=0.9843, val_loss=1.2034, val_acc=0.7481
- 一般化落差: at_best=-1.0405, at_last=-1.1705

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0006
- train_acc_slope: 0.0004
- val_loss_slope: 0.0222
- val_acc_slope: 0.0004

## 過擬合分析
- 判定: 是 (score=2)
- 訊號: early_best=False, loss_rebound=True, gap_large=False, acc_drop=True
- 附註: best_epoch_ratio=0.69

## 設定摘要
- lr: 0.0005
- weight_decay: 0.0001
- seed: 42
- use_norm: True
- balance_by_class: True
- amplify_hard_negative: True
- hard_negative_factor: 1.5
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0