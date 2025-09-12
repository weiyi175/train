# 訓練報告
- 模型: tcn_attention  | 分割: short  | 裝置: cuda  | 參數量: 473091
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 23
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 23

## 核心指標
- 最佳 (epoch 23): train_loss=0.0745, train_acc=0.9628, val_loss=0.8922, val_acc=0.7557
- 最終 (epoch 23): train_loss=0.0745, train_acc=0.9628, val_loss=0.8922, val_acc=0.7557
- 一般化落差: at_best=-0.8177, at_last=-0.8177

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0104
- train_acc_slope: 0.0044
- val_loss_slope: -0.1028
- val_acc_slope: 0.0172

## 過擬合分析
- 判定: 否 (score=0)
- 訊號: early_best=False, loss_rebound=False, gap_large=False, acc_drop=False
- 附註: best_epoch_ratio=1.00

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