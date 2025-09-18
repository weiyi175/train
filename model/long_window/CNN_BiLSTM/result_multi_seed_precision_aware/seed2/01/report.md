# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed2 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 2 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7161
- F1: 0.6190
- Recall: 0.6190
- Precision: 0.6190
- Composite Score: 0.6384 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6384 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 65
- FP: 40
- FN: 40
- TN: 111

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 55: auc=0.7369, f1=0.6508, recall=0.7736, precision=0.5616, score=0.7294, precisionAware=0.6234  TP:41 FP:32 FN:12 TN:43
- epoch 53: auc=0.7517, f1=0.6609, recall=0.7170, precision=0.6129, score=0.7071, precisionAware=0.6551  TP:38 FP:24 FN:15 TN:51
- epoch 50: auc=0.7331, f1=0.6393, recall=0.7358, precision=0.5652, score=0.7063, precisionAware=0.6210  TP:39 FP:30 FN:14 TN:45
- epoch 48: auc=0.7270, f1=0.6230, recall=0.7170, precision=0.5507, score=0.6908, precisionAware=0.6077  TP:38 FP:31 FN:15 TN:44

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 26: auc=0.7203, f1=0.5618, recall=0.4717, precision=0.6944, precisionAware=0.6598, composite=0.5484  TP:25 FP:11 FN:28 TN:64
- epoch 53: auc=0.7517, f1=0.6609, recall=0.7170, precision=0.6129, precisionAware=0.6551, composite=0.7071  TP:38 FP:24 FN:15 TN:51
- epoch 58: auc=0.7582, f1=0.6226, recall=0.6226, precision=0.6226, precisionAware=0.6498, composite=0.6498  TP:33 FP:20 FN:20 TN:55
- epoch 49: auc=0.7600, f1=0.5591, recall=0.4906, precision=0.6500, precisionAware=0.6447, composite=0.5650  TP:26 FP:14 FN:27 TN:61
