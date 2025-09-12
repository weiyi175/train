# 訓練報告
- 模型: tcn_attention  | 分割: long  | 裝置: cuda  | 參數量: 991043
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 10
- 資料: N=1276 T=75 F=36  | 批次: 128  | epoch: 10

## 核心指標
- 最佳 (epoch 9): train_loss=0.3226, train_acc=0.8962, val_loss=0.9820, val_acc=0.6353
- 最終 (epoch 10): train_loss=0.2945, train_acc=0.9089, val_loss=0.9907, val_acc=0.6314
- 一般化落差: at_best=-0.6594, at_last=-0.6962

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0093
- train_acc_slope: 0.0032
- val_loss_slope: -0.0226
- val_acc_slope: 0.0009

## 學習率建議
- 建議: 維持  | 當前 lr: 3e-05 
- 理由: val_loss 持續下降，暫時維持目前 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 9: train_loss=0.3226, train_acc=0.8962, val_loss=0.9820, val_acc=0.6353
2. epoch 5: train_loss=0.3204, train_acc=0.9021, val_loss=0.9827, val_acc=0.6471
3. epoch 10: train_loss=0.2945, train_acc=0.9089, val_loss=0.9907, val_acc=0.6314
4. epoch 7: train_loss=0.2873, train_acc=0.9138, val_loss=1.0348, val_acc=0.6039

## 過擬合分析
- 判定: 否 (score=0)
- 訊號: early_best=False, loss_rebound=False, gap_large=False, acc_drop=False
- 附註: best_epoch_ratio=0.90

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