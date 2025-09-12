# 訓練報告
- 模型: tcn_attention  | 分割: long  | 裝置: cuda  | 參數量: 991043
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 17
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 17
- NOTE: config.epochs=40 but detected 17 epochs in logs/checkpoints

## 核心指標
- 最佳 (epoch 1): train_loss=0.2085, train_acc=0.9158, val_loss=0.8236, val_acc=0.6588
- 最終 (epoch 17): train_loss=0.1513, train_acc=0.9334, val_loss=0.9317, val_acc=0.6588
- 一般化落差: at_best=-0.6151, at_last=-0.7804

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: 0.0007
- train_acc_slope: -0.0010
- val_loss_slope: -0.0006
- val_acc_slope: -0.0004

## 學習率建議
- 建議: 維持  | 當前 lr: 3e-05 
- 理由: val_loss 持續下降，暫時維持目前 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 1: train_loss=0.2085, train_acc=0.9158, val_loss=0.8236, val_acc=0.6588
2. epoch 2: train_loss=0.1793, train_acc=0.9334, val_loss=0.8381, val_acc=0.6784
3. epoch 3: train_loss=0.1871, train_acc=0.9246, val_loss=0.8471, val_acc=0.6706
4. epoch 5: train_loss=0.1924, train_acc=0.9187, val_loss=0.8696, val_acc=0.6824

## 過擬合分析
- 判定: 否 (score=1)
- 訊號: early_best=True, loss_rebound=False, gap_large=False, acc_drop=False
- 附註: best_epoch_ratio=0.06

## 設定摘要
- lr: 3e-05
- weight_decay: 0.0001
- seed: 42
- use_norm: True
- balance_by_class: False
- amplify_hard_negative: False
- hard_negative_factor: 1.5
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0