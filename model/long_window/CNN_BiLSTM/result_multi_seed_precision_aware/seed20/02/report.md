# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed20 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 20 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7398
- F1: 0.5556
- Recall: 0.4762
- Precision: 0.6667
- Composite Score: 0.5527 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6480 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 50
- FP: 25
- FN: 55
- TN: 126

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 57: auc=0.7421, f1=0.6281, recall=0.7170, precision=0.5588, score=0.6953, precisionAware=0.6163  TP:38 FP:30 FN:15 TN:45
- epoch 59: auc=0.7570, f1=0.6207, recall=0.6792, precision=0.5714, score=0.6772, precisionAware=0.6233  TP:36 FP:27 FN:17 TN:48
- epoch 44: auc=0.7306, f1=0.6261, recall=0.6792, precision=0.5806, score=0.6736, precisionAware=0.6243  TP:36 FP:26 FN:17 TN:49
- epoch 53: auc=0.7509, f1=0.6306, recall=0.6604, precision=0.6034, score=0.6696, precisionAware=0.6411  TP:35 FP:23 FN:18 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 27: auc=0.7545, f1=0.6476, recall=0.6415, precision=0.6538, precisionAware=0.6721, composite=0.6659  TP:34 FP:18 FN:19 TN:57
- epoch 49: auc=0.7665, f1=0.6355, recall=0.6415, precision=0.6296, precisionAware=0.6588, composite=0.6647  TP:34 FP:20 FN:19 TN:55
- epoch 46: auc=0.7633, f1=0.6286, recall=0.6226, precision=0.6346, precisionAware=0.6585, composite=0.6525  TP:33 FP:19 FN:20 TN:56
- epoch 43: auc=0.7394, f1=0.6139, recall=0.5849, precision=0.6458, precisionAware=0.6549, composite=0.6245  TP:31 FP:17 FN:22 TN:58
