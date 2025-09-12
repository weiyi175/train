# 訓練報告
- 模型: gcn_tcn  | 分割: short  | 裝置: cuda  | 參數量: 94930
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 80

## 核心指標
- 最佳 (epoch 56): train_loss=0.2633, train_acc=0.8985, val_loss=0.5256, val_acc=0.7615
- 最終 (epoch 80): train_loss=0.1787, train_acc=0.9247, val_loss=0.5673, val_acc=0.7634
- 一般化落差: at_best=-0.2623, at_last=-0.3886

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0028
- train_acc_slope: 0.0000
- val_loss_slope: 0.0030
- val_acc_slope: -0.0015

## 學習率建議
- 建議: 調低  | 當前 lr: 0.0001 
- 理由: val_loss 出現回彈或上升，可能步長過大或不穩定，建議調低 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 56: train_loss=0.2633, train_acc=0.8985, val_loss=0.5256, val_acc=0.7615
2. epoch 66: train_loss=0.1993, train_acc=0.9280, val_loss=0.5260, val_acc=0.7672
3. epoch 78: train_loss=0.1866, train_acc=0.9218, val_loss=0.5294, val_acc=0.7824
4. epoch 67: train_loss=0.2187, train_acc=0.9152, val_loss=0.5304, val_acc=0.7748

## 過擬合分析
- 判定: 否 (score=1)
- 訊號: early_best=False, loss_rebound=True, gap_large=False, acc_drop=False
- 附註: best_epoch_ratio=0.70

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