# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed1 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 1 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6619
- F1: 0.6223
- Recall: 0.6359
- Precision: 0.6094
- Composite Score: 0.6370 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6238 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 117
- FP: 75
- FN: 67
- TN: 95

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 48: auc=0.7114, f1=0.6207, recall=0.6792, precision=0.5714, score=0.6681, precisionAware=0.6142  TP:36 FP:27 FN:17 TN:48
- epoch 50: auc=0.7114, f1=0.6000, recall=0.6792, precision=0.5373, score=0.6619, precisionAware=0.5909  TP:36 FP:31 FN:17 TN:44
- epoch 51: auc=0.7190, f1=0.5983, recall=0.6604, precision=0.5469, score=0.6535, precisionAware=0.5967  TP:35 FP:29 FN:18 TN:46
- epoch 54: auc=0.7268, f1=0.6055, recall=0.6226, precision=0.5893, score=0.6383, precisionAware=0.6217  TP:33 FP:23 FN:20 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 54: auc=0.7268, f1=0.6055, recall=0.6226, precision=0.5893, precisionAware=0.6217, composite=0.6383  TP:33 FP:23 FN:20 TN:52
- epoch 46: auc=0.6991, f1=0.5825, recall=0.5660, precision=0.6000, precisionAware=0.6146, composite=0.5976  TP:30 FP:20 FN:23 TN:55
- epoch 53: auc=0.7248, f1=0.6000, recall=0.6226, precision=0.5789, precisionAware=0.6144, composite=0.6363  TP:33 FP:24 FN:20 TN:51
- epoch 48: auc=0.7114, f1=0.6207, recall=0.6792, precision=0.5714, precisionAware=0.6142, composite=0.6681  TP:36 FP:27 FN:17 TN:48
