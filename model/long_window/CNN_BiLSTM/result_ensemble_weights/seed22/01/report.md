# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed22 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 22 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6792
- F1: 0.6701
- Recall: 0.7011
- Precision: 0.6418
- Composite Score: 0.6874 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6578 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 129
- FP: 72
- FN: 55
- TN: 98

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 52: auc=0.7238, f1=0.6519, recall=0.8302, precision=0.5366, score=0.7554, precisionAware=0.6086  TP:44 FP:38 FN:9 TN:37
- epoch 47: auc=0.7197, f1=0.6471, recall=0.8302, precision=0.5301, score=0.7532, precisionAware=0.6031  TP:44 FP:39 FN:9 TN:36
- epoch 48: auc=0.7288, f1=0.6418, recall=0.8113, precision=0.5309, score=0.7440, precisionAware=0.6037  TP:43 FP:38 FN:10 TN:37
- epoch 53: auc=0.7293, f1=0.6412, recall=0.7925, precision=0.5385, score=0.7345, precisionAware=0.6075  TP:42 FP:36 FN:11 TN:39

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 55: auc=0.7424, f1=0.6555, recall=0.7358, precision=0.5909, precisionAware=0.6406, composite=0.7130  TP:39 FP:27 FN:14 TN:48
- epoch 4: auc=0.6091, f1=0.0370, recall=0.0189, precision=1.0000, precisionAware=0.6329, composite=0.1424  TP:1 FP:0 FN:52 TN:75
- epoch 60: auc=0.7497, f1=0.6271, recall=0.6981, precision=0.5692, precisionAware=0.6227, composite=0.6871  TP:37 FP:28 FN:16 TN:47
- epoch 50: auc=0.7316, f1=0.6281, recall=0.7170, precision=0.5588, precisionAware=0.6142, composite=0.6932  TP:38 FP:30 FN:15 TN:45
