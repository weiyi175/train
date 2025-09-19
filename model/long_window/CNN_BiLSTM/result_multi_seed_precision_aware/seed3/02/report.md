# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed3 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 3 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6718
- F1: 0.6049
- Recall: 0.6033
- Precision: 0.6066
- Composite Score: 0.6175 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6191 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 111
- FP: 72
- FN: 73
- TN: 98

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 32: auc=0.6860, f1=0.6515, recall=0.8113, precision=0.5443, score=0.7383, precisionAware=0.6048  TP:43 FP:36 FN:10 TN:39
- epoch 55: auc=0.7628, f1=0.6723, recall=0.7547, precision=0.6061, score=0.7316, precisionAware=0.6573  TP:40 FP:26 FN:13 TN:49
- epoch 56: auc=0.7369, f1=0.6612, recall=0.7547, precision=0.5882, score=0.7231, precisionAware=0.6398  TP:40 FP:28 FN:13 TN:47
- epoch 43: auc=0.7210, f1=0.6250, recall=0.7547, precision=0.5333, score=0.7091, precisionAware=0.5984  TP:40 FP:35 FN:13 TN:40

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 47: auc=0.7587, f1=0.6545, recall=0.6792, precision=0.6316, precisionAware=0.6639, composite=0.6877  TP:36 FP:21 FN:17 TN:54
- epoch 44: auc=0.7507, f1=0.6481, recall=0.6604, precision=0.6364, precisionAware=0.6628, composite=0.6748  TP:35 FP:20 FN:18 TN:55
- epoch 50: auc=0.7499, f1=0.6545, recall=0.6792, precision=0.6316, precisionAware=0.6621, composite=0.6860  TP:36 FP:21 FN:17 TN:54
- epoch 54: auc=0.7688, f1=0.6286, recall=0.6226, precision=0.6346, precisionAware=0.6596, composite=0.6537  TP:33 FP:19 FN:20 TN:56
