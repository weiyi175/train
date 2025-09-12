# 訓練報告
- 模型: tcn_attention  | 分割: long  | 裝置: cuda  | 參數量: 991043
- 資料: N=1276 T=75 F=36  | 批次: 256  | epoch: 60
- 資料: N=1276 T=75 F=36  | 批次: 256  | epoch: 60

## 核心指標
- 最佳 (epoch 15): train_loss=0.6085, train_acc=0.6582, val_loss=0.6196, val_acc=0.6510
- 最終 (epoch 60): train_loss=0.1558, train_acc=0.9461, val_loss=0.9255, val_acc=0.6353
- 一般化落差: at_best=-0.0111, at_last=-0.7697

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0071
- train_acc_slope: 0.0040
- val_loss_slope: -0.0219
- val_acc_slope: 0.0078

## 學習率建議
- 建議: 維持  | 當前 lr: 5e-05 
- 理由: val_loss 持續下降，暫時維持目前 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 15: train_loss=0.6085, train_acc=0.6582, val_loss=0.6196, val_acc=0.6510
2. epoch 11: train_loss=0.6402, train_acc=0.6180, val_loss=0.6257, val_acc=0.6431
3. epoch 20: train_loss=0.5635, train_acc=0.6797, val_loss=0.6265, val_acc=0.6353
4. epoch 21: train_loss=0.5679, train_acc=0.6895, val_loss=0.6317, val_acc=0.6314

## 過擬合分析
- 判定: 否 (score=1)
- 訊號: early_best=True, loss_rebound=False, gap_large=False, acc_drop=False
- 附註: best_epoch_ratio=0.25

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