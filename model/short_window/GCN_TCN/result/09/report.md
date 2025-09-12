# 訓練報告
- 模型: gcn_tcn  | 分割: short  | 裝置: cuda  | 參數量: 94930
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 300

## 核心指標
- 最佳 (epoch 66): train_loss=0.1990, train_acc=0.9295, val_loss=0.5256, val_acc=0.7691
- 最終 (epoch 300): train_loss=0.0457, train_acc=0.9828, val_loss=0.7882, val_acc=0.7939
- 一般化落差: at_best=-0.3266, at_last=-0.7425

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: 0.0002
- train_acc_slope: -0.0001
- val_loss_slope: 0.0009
- val_acc_slope: -0.0006

## 學習率建議
- 建議: 調低  | 當前 lr: 0.0001 
- 理由: val_loss 出現回彈或上升，可能步長過大或不穩定，建議調低 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 66: train_loss=0.1990, train_acc=0.9295, val_loss=0.5256, val_acc=0.7691
2. epoch 56: train_loss=0.2631, train_acc=0.8980, val_loss=0.5266, val_acc=0.7557
3. epoch 78: train_loss=0.1853, train_acc=0.9233, val_loss=0.5286, val_acc=0.7824
4. epoch 67: train_loss=0.2189, train_acc=0.9137, val_loss=0.5300, val_acc=0.7672

## 過擬合分析
- 判定: 是 (score=2)
- 訊號: early_best=True, loss_rebound=True, gap_large=False, acc_drop=False
- 附註: best_epoch_ratio=0.22

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