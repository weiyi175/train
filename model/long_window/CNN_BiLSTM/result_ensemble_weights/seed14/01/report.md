# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed14 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 14 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6737
- F1: 0.6472
- Recall: 0.6630
- Precision: 0.6321
- Composite Score: 0.6604 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6450 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 122
- FP: 71
- FN: 62
- TN: 99

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.4249, f1=0.5856, recall=1.0000, precision=0.4141, score=0.7607, precisionAware=0.4677  TP:53 FP:75 FN:0 TN:0
- epoch 2: auc=0.4528, f1=0.5810, recall=0.9811, precision=0.4127, score=0.7554, precisionAware=0.4712  TP:52 FP:74 FN:1 TN:1
- epoch 53: auc=0.6881, f1=0.6066, recall=0.6981, precision=0.5362, score=0.6686, precisionAware=0.5877  TP:37 FP:32 FN:16 TN:43
- epoch 52: auc=0.6848, f1=0.5920, recall=0.6981, precision=0.5139, score=0.6636, precisionAware=0.5715  TP:37 FP:35 FN:16 TN:40

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 5: auc=0.6267, f1=0.0727, recall=0.0377, precision=1.0000, precisionAware=0.6472, composite=0.1660  TP:2 FP:0 FN:51 TN:75
- epoch 8: auc=0.6131, f1=0.0727, recall=0.0377, precision=1.0000, precisionAware=0.6444, composite=0.1633  TP:2 FP:0 FN:51 TN:75
- epoch 6: auc=0.6231, f1=0.0370, recall=0.0189, precision=1.0000, precisionAware=0.6357, composite=0.1452  TP:1 FP:0 FN:52 TN:75
- epoch 7: auc=0.6158, f1=0.0370, recall=0.0189, precision=1.0000, precisionAware=0.6343, composite=0.1437  TP:1 FP:0 FN:52 TN:75
