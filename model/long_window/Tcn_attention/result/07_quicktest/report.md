# 訓練報告
- 模型: tcn_attention  | 分割: long  | 裝置: cuda  | 參數量: 991043
- 資料: N=1276 T=75 F=36  | 批次: 32  | epoch: 30
- 資料: N=1276 T=75 F=36  | 批次: 32  | epoch: 30

## 核心指標
- 最佳 (epoch 6): train_loss=0.5371, train_acc=0.7297, val_loss=0.6504, val_acc=0.6471
- 最終 (epoch 30): train_loss=0.0870, train_acc=0.9638, val_loss=1.4191, val_acc=0.7059
- 一般化落差: at_best=-0.1133, at_last=-1.3321

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0059
- train_acc_slope: 0.0023
- val_loss_slope: -0.0883
- val_acc_slope: 0.0179

## 學習率建議
- 建議: 維持  | 當前 lr: 0.0001 
- 理由: val_loss 持續下降，暫時維持目前 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 6: train_loss=0.5371, train_acc=0.7297, val_loss=0.6504, val_acc=0.6471
2. epoch 4: train_loss=0.6020, train_acc=0.6641, val_loss=0.6524, val_acc=0.6392
3. epoch 2: train_loss=0.6521, train_acc=0.6151, val_loss=0.6689, val_acc=0.5608
4. epoch 1: train_loss=0.6786, train_acc=0.5769, val_loss=0.6743, val_acc=0.5451

## 過擬合分析
- 判定: 否 (score=1)
- 訊號: early_best=True, loss_rebound=False, gap_large=False, acc_drop=False
- 附註: best_epoch_ratio=0.20

## 設定摘要
- lr: 0.0001
- weight_decay: 0.0001
- seed: 7
- use_norm: True
- balance_by_class: True
- amplify_hard_negative: False
- hard_negative_factor: 1.0
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0