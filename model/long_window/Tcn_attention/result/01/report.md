# 訓練報告
- 模型: tcn_attention  | 分割: long  | 裝置: cuda  | 參數量: 991043
- 資料: N=1276 T=75 F=36  | 批次: 64  | epoch: 1
- 資料: N=1276 T=75 F=36  | 批次: 64  | epoch: 1

## 核心指標
- 最佳 (epoch 1): train_loss=0.6773, train_acc=0.5602, val_loss=0.6711, val_acc=0.6314
- 最終 (epoch 1): train_loss=0.6773, train_acc=0.5602, val_loss=0.6711, val_acc=0.6314
- 一般化落差: at_best=0.0062, at_last=0.0062

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: 0.0000
- train_acc_slope: 0.0000
- val_loss_slope: 0.0000
- val_acc_slope: 0.0000

## 學習率建議
- 建議: 調低  | 當前 lr: 0.0001 
- 理由: val_loss 近似平穩（plateau），建議調低 learning rate 以精細化收斂。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 1: train_loss=0.6773, train_acc=0.5602, val_loss=0.6711, val_acc=0.6314

## 過擬合分析
- 判定: 否 (score=0)
- 訊號: early_best=False, loss_rebound=False, gap_large=False, acc_drop=False
- 附註: best_epoch_ratio=1.00

## 設定摘要
- lr: 0.0001
- weight_decay: 0.0001
- seed: 42
- use_norm: True
- balance_by_class: True
- amplify_hard_negative: True
- hard_negative_factor: 1.5
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0