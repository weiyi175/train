# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed41 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 41 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7255
- F1: 0.6684
- Recall: 0.6793
- Precision: 0.6579
- Composite Score: 0.6853 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6746 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 125
- FP: 65
- FN: 59
- TN: 105

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 51: auc=0.7255, f1=0.6452, recall=0.7547, precision=0.5634, score=0.7160, precisionAware=0.6203  TP:40 FP:31 FN:13 TN:44
- epoch 46: auc=0.7026, f1=0.6212, recall=0.7736, precision=0.5190, score=0.7137, precisionAware=0.5864  TP:41 FP:38 FN:12 TN:37
- epoch 52: auc=0.7230, f1=0.6290, recall=0.7358, precision=0.5493, score=0.7012, precisionAware=0.6080  TP:39 FP:32 FN:14 TN:43
- epoch 55: auc=0.7263, f1=0.6240, recall=0.7358, precision=0.5417, score=0.7004, precisionAware=0.6033  TP:39 FP:33 FN:14 TN:42

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 49: auc=0.7348, f1=0.6296, recall=0.6415, precision=0.6182, precisionAware=0.6449, composite=0.6566  TP:34 FP:21 FN:19 TN:54
- epoch 50: auc=0.7341, f1=0.6491, recall=0.6981, precision=0.6066, precisionAware=0.6448, composite=0.6906  TP:37 FP:24 FN:16 TN:51
- epoch 56: auc=0.7321, f1=0.6496, recall=0.7170, precision=0.5938, precisionAware=0.6382, composite=0.6998  TP:38 FP:26 FN:15 TN:49
- epoch 54: auc=0.7263, f1=0.6441, recall=0.7170, precision=0.5846, precisionAware=0.6308, composite=0.6970  TP:38 FP:27 FN:15 TN:48
