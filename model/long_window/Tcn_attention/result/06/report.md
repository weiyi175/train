# 訓練報告
- 模型: tcn_attention  | 分割: long  | 裝置: cuda  | 參數量: 991043
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 10
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 10

## 核心指標
- 最佳 (epoch 5): train_loss=0.5043, train_acc=0.7845, val_loss=0.7559, val_acc=0.6235
- 最終 (epoch 10): train_loss=0.4681, train_acc=0.7855, val_loss=0.8479, val_acc=0.5765
- 一般化落差: at_best=-0.2516, at_last=-0.3798

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0063
- train_acc_slope: -0.0004
- val_loss_slope: -0.0240
- val_acc_slope: -0.0035

## 學習率建議
- 建議: 維持  | 當前 lr: 5e-05 
- 理由: val_loss 持續下降，暫時維持目前 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 5: train_loss=0.5043, train_acc=0.7845, val_loss=0.7559, val_acc=0.6235
2. epoch 9: train_loss=0.4777, train_acc=0.8002, val_loss=0.7792, val_acc=0.6078
3. epoch 6: train_loss=0.4850, train_acc=0.7875, val_loss=0.8013, val_acc=0.6000
4. epoch 8: train_loss=0.4688, train_acc=0.7924, val_loss=0.8120, val_acc=0.5922

## 過擬合分析
- 判定: 否 (score=1)
- 訊號: early_best=False, loss_rebound=False, gap_large=False, acc_drop=True
- 附註: best_epoch_ratio=0.50

## 設定摘要
- lr: 5e-05
- weight_decay: 0.0001
- seed: 42
- use_norm: True
- balance_by_class: False
- amplify_hard_negative: False
- hard_negative_factor: 1.5
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0