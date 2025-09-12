# 訓練報告
- 模型: tcn_attention  | 分割: long  | 裝置: cuda  | 參數量: 991043
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 60
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 60

## 核心指標
- 最佳 (epoch 19): train_loss=0.5635, train_acc=0.6944, val_loss=0.6038, val_acc=0.6392
- 最終 (epoch 60): train_loss=0.1554, train_acc=0.9383, val_loss=1.0175, val_acc=0.6235
- 一般化落差: at_best=-0.0403, at_last=-0.8621

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0097
- train_acc_slope: 0.0049
- val_loss_slope: 0.0269
- val_acc_slope: -0.0061

## 學習率建議
- 建議: 調低  | 當前 lr: 3e-05 
- 理由: val_loss 上升或震盪，可能步長過大或不穩定，建議調低 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 19: train_loss=0.5635, train_acc=0.6944, val_loss=0.6038, val_acc=0.6392
2. epoch 16: train_loss=0.5829, train_acc=0.6856, val_loss=0.6044, val_acc=0.6510
3. epoch 24: train_loss=0.5100, train_acc=0.7356, val_loss=0.6056, val_acc=0.6588
4. epoch 28: train_loss=0.4749, train_acc=0.7806, val_loss=0.6090, val_acc=0.6588

## 過擬合分析
- 判定: 是 (score=2)
- 訊號: early_best=True, loss_rebound=True, gap_large=False, acc_drop=False
- 附註: best_epoch_ratio=0.32

## 設定摘要
- lr: 3e-05
- weight_decay: 0.0001
- seed: 42
- use_norm: True
- balance_by_class: False
- amplify_hard_negative: False
- hard_negative_factor: 1.0
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0