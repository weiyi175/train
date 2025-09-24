# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_slipce/seed4 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 4 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7233
- F1: 0.7016
- Recall: 0.7283
- Precision: 0.6768
- Composite Score: 0.7193 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6935 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 134
- FP: 64
- FN: 50
- TN: 106

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 33: auc=0.7716, f1=0.6842, recall=0.7358, precision=0.6393, score=0.7275, precisionAware=0.6792  TP:39 FP:22 FN:14 TN:53
- epoch 31: auc=0.7673, f1=0.6842, recall=0.7358, precision=0.6393, score=0.7266, precisionAware=0.6784  TP:39 FP:22 FN:14 TN:53
- epoch 49: auc=0.7376, f1=0.6723, recall=0.7547, precision=0.6061, score=0.7266, precisionAware=0.6522  TP:40 FP:26 FN:13 TN:49
- epoch 51: auc=0.7535, f1=0.6783, recall=0.7358, precision=0.6290, score=0.7221, precisionAware=0.6687  TP:39 FP:23 FN:14 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 33: auc=0.7716, f1=0.6842, recall=0.7358, precision=0.6393, precisionAware=0.6792, composite=0.7275  TP:39 FP:22 FN:14 TN:53
- epoch 31: auc=0.7673, f1=0.6842, recall=0.7358, precision=0.6393, precisionAware=0.6784, composite=0.7266  TP:39 FP:22 FN:14 TN:53
- epoch 30: auc=0.7728, f1=0.6408, recall=0.6226, precision=0.6600, precisionAware=0.6768, composite=0.6581  TP:33 FP:17 FN:20 TN:58
- epoch 36: auc=0.7429, f1=0.6542, recall=0.6604, precision=0.6481, precisionAware=0.6689, composite=0.6750  TP:35 FP:19 FN:18 TN:56
