# 訓練報告
- 模型: tcn_attention  | 分割: long  | 裝置: cuda  | 參數量: 991043
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 60
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 60

## 核心指標
- 最佳 (epoch 15): train_loss=0.5357, train_acc=0.7316, val_loss=0.6038, val_acc=0.6431
- 最終 (epoch 60): train_loss=0.0713, train_acc=0.9716, val_loss=1.4798, val_acc=0.6078
- 一般化落差: at_best=-0.0681, at_last=-1.4084

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0061
- train_acc_slope: 0.0022
- val_loss_slope: 0.0365
- val_acc_slope: -0.0031

## 學習率建議
- 建議: 調低  | 當前 lr: 5e-05 
- 理由: val_loss 上升或震盪，可能步長過大或不穩定，建議調低 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 15: train_loss=0.5357, train_acc=0.7316, val_loss=0.6038, val_acc=0.6431
2. epoch 18: train_loss=0.4738, train_acc=0.7747, val_loss=0.6070, val_acc=0.6627
3. epoch 16: train_loss=0.5262, train_acc=0.7297, val_loss=0.6072, val_acc=0.6588
4. epoch 10: train_loss=0.5960, train_acc=0.6582, val_loss=0.6109, val_acc=0.6235

## 過擬合分析
- 判定: 是 (score=3)
- 訊號: early_best=True, loss_rebound=True, gap_large=False, acc_drop=True
- 附註: best_epoch_ratio=0.25

## 設定摘要
- lr: 5e-05
- weight_decay: 0.0001
- seed: 42
- use_norm: True
- balance_by_class: False
- amplify_hard_negative: False
- hard_negative_factor: 1.0
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0