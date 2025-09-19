# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed19 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 19 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6745
- F1: 0.5970
- Recall: 0.5435
- Precision: 0.6623
- Composite Score: 0.5858 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6451 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 100
- FP: 51
- FN: 84
- TN: 119

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 35: auc=0.7585, f1=0.6885, recall=0.7925, precision=0.6087, score=0.7545, precisionAware=0.6626  TP:42 FP:27 FN:11 TN:48
- epoch 39: auc=0.7406, f1=0.6667, recall=0.8113, precision=0.5658, score=0.7538, precisionAware=0.6310  TP:43 FP:33 FN:10 TN:42
- epoch 31: auc=0.7472, f1=0.6949, recall=0.7736, precision=0.6308, score=0.7447, precisionAware=0.6733  TP:41 FP:24 FN:12 TN:51
- epoch 45: auc=0.7940, f1=0.6842, recall=0.7358, precision=0.6393, score=0.7320, precisionAware=0.6837  TP:39 FP:22 FN:14 TN:53

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 49: auc=0.8010, f1=0.6458, recall=0.5849, precision=0.7209, precisionAware=0.7144, composite=0.6464  TP:31 FP:12 FN:22 TN:63
- epoch 48: auc=0.7884, f1=0.6731, recall=0.6604, precision=0.6863, precisionAware=0.7027, composite=0.6898  TP:35 FP:16 FN:18 TN:59
- epoch 29: auc=0.7769, f1=0.6535, recall=0.6226, precision=0.6875, precisionAware=0.6952, composite=0.6627  TP:33 FP:15 FN:20 TN:60
- epoch 47: auc=0.7967, f1=0.5934, recall=0.5094, precision=0.7105, precisionAware=0.6926, composite=0.5921  TP:27 FP:11 FN:26 TN:64
