# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed31 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 31 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6522
- F1: 0.6683
- Recall: 0.7609
- Precision: 0.5957
- Composite Score: 0.7113 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6288 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 140
- FP: 95
- FN: 44
- TN: 75

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 52: auc=0.7021, f1=0.6277, recall=0.8113, precision=0.5119, score=0.7344, precisionAware=0.5847  TP:43 FP:41 FN:10 TN:34
- epoch 54: auc=0.7064, f1=0.6222, recall=0.7925, precision=0.5122, score=0.7242, precisionAware=0.5840  TP:42 FP:40 FN:11 TN:35
- epoch 59: auc=0.6911, f1=0.6176, recall=0.7925, precision=0.5060, score=0.7197, precisionAware=0.5765  TP:42 FP:41 FN:11 TN:34
- epoch 53: auc=0.6976, f1=0.6043, recall=0.7925, precision=0.4884, score=0.7170, precisionAware=0.5650  TP:42 FP:44 FN:11 TN:31

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 3: auc=0.6068, f1=0.1071, recall=0.0566, precision=1.0000, precisionAware=0.6535, composite=0.1818  TP:3 FP:0 FN:50 TN:75
- epoch 2: auc=0.6003, f1=0.0727, recall=0.0377, precision=1.0000, precisionAware=0.6419, composite=0.1607  TP:2 FP:0 FN:51 TN:75
- epoch 56: auc=0.7117, f1=0.6325, recall=0.6981, precision=0.5781, precisionAware=0.6211, composite=0.6811  TP:37 FP:27 FN:16 TN:48
- epoch 57: auc=0.7114, f1=0.6281, recall=0.7170, precision=0.5588, precisionAware=0.6101, composite=0.6892  TP:38 FP:30 FN:15 TN:45
