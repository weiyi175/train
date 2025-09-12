# 訓練報告
- 模型: tcn_attention  | 分割: long  | 裝置: cuda  | 參數量: 991043
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 60
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 60

## 核心指標
- 最佳 (epoch 20): train_loss=0.4671, train_acc=0.7816, val_loss=0.6140, val_acc=0.6510
- 最終 (epoch 60): train_loss=0.0913, train_acc=0.9608, val_loss=1.5102, val_acc=0.6275
- 一般化落差: at_best=-0.1469, at_last=-1.4188

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0024
- train_acc_slope: 0.0007
- val_loss_slope: -0.0024
- val_acc_slope: 0.0022

## 學習率建議
- 建議: 維持  | 當前 lr: 5e-05 
- 理由: val_loss 持續下降，暫時維持目前 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 20: train_loss=0.4671, train_acc=0.7816, val_loss=0.6140, val_acc=0.6510
2. epoch 10: train_loss=0.6192, train_acc=0.6396, val_loss=0.6173, val_acc=0.6510
3. epoch 13: train_loss=0.5794, train_acc=0.6807, val_loss=0.6183, val_acc=0.6392
4. epoch 18: train_loss=0.5065, train_acc=0.7689, val_loss=0.6202, val_acc=0.6471

## 過擬合分析
- 判定: 是 (score=2)
- 訊號: early_best=True, loss_rebound=False, gap_large=False, acc_drop=True
- 附註: best_epoch_ratio=0.33

## 設定摘要
- lr: 5e-05
- weight_decay: 0.0001
- seed: 42
- use_norm: True
- balance_by_class: True
- amplify_hard_negative: False
- hard_negative_factor: 1.0
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0