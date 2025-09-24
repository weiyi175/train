# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_center/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_center/seed10 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 10 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6990
- F1: 0.5467
- Recall: 0.4457
- Precision: 0.7069
- Composite Score: 0.5266 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6572 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 82
- FP: 34
- FN: 102
- TN: 136

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 50: auc=0.7683, f1=0.6942, recall=0.7778, precision=0.6269, score=0.7508, precisionAware=0.6754  TP:42 FP:25 FN:12 TN:49
- epoch 42: auc=0.7643, f1=0.6972, recall=0.7037, precision=0.6909, score=0.7139, precisionAware=0.7075  TP:38 FP:17 FN:16 TN:57
- epoch 27: auc=0.7953, f1=0.7200, recall=0.6667, precision=0.7826, score=0.7084, precisionAware=0.7664  TP:36 FP:10 FN:18 TN:64
- epoch 25: auc=0.7975, f1=0.7059, recall=0.6667, precision=0.7500, score=0.7046, precisionAware=0.7463  TP:36 FP:12 FN:18 TN:62

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 27: auc=0.7953, f1=0.7200, recall=0.6667, precision=0.7826, precisionAware=0.7664, composite=0.7084  TP:36 FP:10 FN:18 TN:64
- epoch 26: auc=0.7760, f1=0.6279, recall=0.5000, precision=0.8438, precisionAware=0.7655, composite=0.5936  TP:27 FP:5 FN:27 TN:69
- epoch 28: auc=0.7695, f1=0.7200, recall=0.6667, precision=0.7826, precisionAware=0.7612, composite=0.7032  TP:36 FP:10 FN:18 TN:64
- epoch 25: auc=0.7975, f1=0.7059, recall=0.6667, precision=0.7500, precisionAware=0.7463, composite=0.7046  TP:36 FP:12 FN:18 TN:62
