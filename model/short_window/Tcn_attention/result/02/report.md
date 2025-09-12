# 訓練報告
- 模型: tcn_attention  | 分割: short  | 裝置: cuda  | 參數量: 473091
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 100
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 100

## 核心指標
- 最佳 (epoch 46): train_loss=0.0378, train_acc=0.9800, val_loss=0.9903, val_acc=0.7767
- 最終 (epoch 100): train_loss=0.0320, train_acc=0.9824, val_loss=1.6005, val_acc=0.7481
- 一般化落差: at_best=-0.9525, at_last=-1.5686

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: 0.0001
- train_acc_slope: -0.0002
- val_loss_slope: 0.0149
- val_acc_slope: 0.0004

## 過擬合分析
- 判定: 是 (score=2)
- 訊號: early_best=False, loss_rebound=True, gap_large=False, acc_drop=True
- 附註: best_epoch_ratio=0.46

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