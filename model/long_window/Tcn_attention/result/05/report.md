# 訓練報告
- 模型: tcn_attention  | 分割: long  | 裝置: cuda  | 參數量: 991043
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 10
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 10

## 核心指標
- 最佳 (epoch 9): train_loss=0.3156, train_acc=0.8972, val_loss=0.9513, val_acc=0.6314
- 最終 (epoch 10): train_loss=0.2943, train_acc=0.9040, val_loss=0.9739, val_acc=0.6275
- 一般化落差: at_best=-0.6357, at_last=-0.6796

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0106
- train_acc_slope: 0.0032
- val_loss_slope: -0.0267
- val_acc_slope: -0.0009

## 學習率建議
- 建議: 維持  | 當前 lr: 5e-05 
- 理由: val_loss 持續下降，暫時維持目前 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 9: train_loss=0.3156, train_acc=0.8972, val_loss=0.9513, val_acc=0.6314
2. epoch 4: train_loss=0.2850, train_acc=0.9148, val_loss=0.9586, val_acc=0.6353
3. epoch 10: train_loss=0.2943, train_acc=0.9040, val_loss=0.9739, val_acc=0.6275
4. epoch 6: train_loss=0.3127, train_acc=0.9030, val_loss=0.9750, val_acc=0.6314

## 過擬合分析
- 判定: 否 (score=0)
- 訊號: early_best=False, loss_rebound=False, gap_large=False, acc_drop=False
- 附註: best_epoch_ratio=0.90

## 設定摘要
- lr: 5e-05
- weight_decay: 0.0005
- seed: 42
- use_norm: True
- balance_by_class: False
- amplify_hard_negative: False
- hard_negative_factor: 1.5
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0