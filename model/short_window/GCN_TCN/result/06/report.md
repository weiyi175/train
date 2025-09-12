# 訓練報告
- 模型: gcn_tcn  | 分割: short  | 裝置: cuda  | 參數量: 94930
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 50

## 核心指標
- 最佳 (epoch 16): train_loss=0.2277, train_acc=0.9066, val_loss=0.5356, val_acc=0.7615
- 最終 (epoch 50): train_loss=0.0744, train_acc=0.9657, val_loss=0.7903, val_acc=0.7653
- 一般化落差: at_best=-0.3079, at_last=-0.7159

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0006
- train_acc_slope: -0.0001
- val_loss_slope: 0.0069
- val_acc_slope: -0.0019

## 學習率建議
- 建議: 調低  | 當前 lr: 0.0008 
- 理由: val_loss 出現回彈或上升，可能步長過大或不穩定，建議調低 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 16: train_loss=0.2277, train_acc=0.9066, val_loss=0.5356, val_acc=0.7615
2. epoch 15: train_loss=0.2475, train_acc=0.8918, val_loss=0.5558, val_acc=0.7462
3. epoch 8: train_loss=0.4242, train_acc=0.8098, val_loss=0.5681, val_acc=0.7061
4. epoch 18: train_loss=0.2244, train_acc=0.8994, val_loss=0.5692, val_acc=0.7767

## 過擬合分析
- 判定: 是 (score=2)
- 訊號: early_best=True, loss_rebound=True, gap_large=False, acc_drop=False
- 附註: best_epoch_ratio=0.32

## 設定摘要
- lr: 0.0008
- weight_decay: 0.0001
- seed: 42
- use_norm: True
- balance_by_class: True
- amplify_hard_negative: True
- hard_negative_factor: 1.5
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0