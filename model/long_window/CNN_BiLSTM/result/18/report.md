# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh035/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 50 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 42 --checkpoint_metric auc --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6577
- F1: 0.5859
- Recall: 0.5652
- Precision: 0.6082
- Composite Score: 0.5899 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6114 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 104
- FP: 67
- FN: 80
- TN: 103

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 49: auc=0.8287, f1=0.7755, recall=0.8507, precision=0.7125, score=0.8238, precisionAware=0.7546  TP:57 FP:23 FN:10 TN:38
- epoch 46: auc=0.8429, f1=0.7857, recall=0.8209, precision=0.7534, score=0.8147, precisionAware=0.7810  TP:55 FP:18 FN:12 TN:43
- epoch 38: auc=0.8101, f1=0.7778, recall=0.8358, precision=0.7273, score=0.8133, precisionAware=0.7590  TP:56 FP:21 FN:11 TN:40
- epoch 42: auc=0.8131, f1=0.7971, recall=0.8209, precision=0.7746, score=0.8122, precisionAware=0.7891  TP:55 FP:16 FN:12 TN:45

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 47: auc=0.8412, f1=0.8062, recall=0.7761, precision=0.8387, precisionAware=0.8295, composite=0.7982  TP:52 FP:10 FN:15 TN:51
- epoch 41: auc=0.8241, f1=0.7288, recall=0.6418, precision=0.8431, precisionAware=0.8050, composite=0.7044  TP:43 FP:8 FN:24 TN:53
- epoch 40: auc=0.8126, f1=0.7786, recall=0.7612, precision=0.7969, precisionAware=0.7945, composite=0.7767  TP:51 FP:13 FN:16 TN:48
- epoch 45: auc=0.8055, f1=0.7520, recall=0.7015, precision=0.8103, precisionAware=0.7919, composite=0.7374  TP:47 FP:11 FN:20 TN:50
