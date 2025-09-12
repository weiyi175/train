# 訓練報告
- 模型: gcn_tcn  | 分割: short  | 裝置: cuda  | 參數量: 94930
- 資料: N=2622 T=30 F=36  | 批次: 256  | epoch: 100

## 核心指標
- 最佳 (epoch 16): train_loss=0.2108, train_acc=0.9099, val_loss=0.5438, val_acc=0.7615
- 最終 (epoch 100): train_loss=0.0382, train_acc=0.9852, val_loss=0.8324, val_acc=0.7748
- 一般化落差: at_best=-0.3330, at_last=-0.7941

## 趨勢 (最後 10 個 epoch 粗略斜率)
- train_loss_slope: -0.0030
- train_acc_slope: 0.0013
- val_loss_slope: -0.0063
- val_acc_slope: 0.0006

## 學習率建議
- 建議: 維持  | 當前 lr: 0.001 
- 理由: val_loss 有下降趨勢，暫時維持目前 learning rate。

## Top 4 最佳 epoch (以 val_loss 為主，val_acc 為輔)
1. epoch 16: train_loss=0.2108, train_acc=0.9099, val_loss=0.5438, val_acc=0.7615
2. epoch 8: train_loss=0.3963, train_acc=0.8232, val_loss=0.5839, val_acc=0.6908
3. epoch 13: train_loss=0.2539, train_acc=0.8894, val_loss=0.6017, val_acc=0.7557
4. epoch 4: train_loss=0.5498, train_acc=0.7140, val_loss=0.6028, val_acc=0.6718

## 過擬合分析
- 判定: 否 (score=1)
- 訊號: early_best=True, loss_rebound=False, gap_large=False, acc_drop=False
- 附註: best_epoch_ratio=0.16

## 設定摘要
- lr: 0.001
- weight_decay: 0.0001
- seed: 42
- use_norm: True
- balance_by_class: True
- amplify_hard_negative: True
- hard_negative_factor: 1.5
- temporal_jitter_frames: 0
- val_ratio: 0.2
- num_workers: 0