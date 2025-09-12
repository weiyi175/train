# 訓練報告
- 模型: tcn_attention  | 分割: long  | 裝置: cuda  | 參數量: 991043
- 資料: N=1276 T=75 F=36  | 批次: 256  | epoch: 100
- 資料: N=1276 T=75 F=36  | 批次: 256  | epoch: 100

## 核心指標
- 最佳 (epoch 14): train_loss=0.5307, train_acc=0.7336, val_loss=0.6363, val_acc=0.6392
- 最終 (epoch 100): train_loss=0.0276, train_acc=0.9843, val_loss=2.1361, val_acc=0.6000
- 一般化落差: at_best=-0.1056, at_last=-2.1085

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0011
- train_acc_slope: 0.0001
- val_loss_slope: 0.0490
- val_acc_slope: -0.0035

## 學習率建議
- 建議: 調低  | 當前 lr: 0.0001 
- 理由: val_loss 上升或震盪，可能步長過大或不穩定，建議調低 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 14: train_loss=0.5307, train_acc=0.7336, val_loss=0.6363, val_acc=0.6392
2. epoch 13: train_loss=0.5640, train_acc=0.6836, val_loss=0.6404, val_acc=0.6471
3. epoch 9: train_loss=0.6103, train_acc=0.6611, val_loss=0.6416, val_acc=0.6510
4. epoch 10: train_loss=0.5988, train_acc=0.6552, val_loss=0.6458, val_acc=0.6431

## 過擬合分析
- 判定: 是 (score=3)
- 訊號: early_best=True, loss_rebound=True, gap_large=False, acc_drop=True
- 附註: best_epoch_ratio=0.14

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