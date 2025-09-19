# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed38 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 38 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6822
- F1: 0.6596
- Recall: 0.6793
- Precision: 0.6410
- Composite Score: 0.6740 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6548 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 125
- FP: 70
- FN: 59
- TN: 100

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 58: auc=0.7137, f1=0.6154, recall=0.7547, precision=0.5195, score=0.7047, precisionAware=0.5871  TP:40 FP:37 FN:13 TN:38
- epoch 54: auc=0.7459, f1=0.6435, recall=0.6981, precision=0.5968, score=0.6913, precisionAware=0.6406  TP:37 FP:25 FN:16 TN:50
- epoch 47: auc=0.7175, f1=0.6230, recall=0.7170, precision=0.5507, score=0.6889, precisionAware=0.6057  TP:38 FP:31 FN:15 TN:44
- epoch 52: auc=0.7326, f1=0.6080, recall=0.7170, precision=0.5278, score=0.6874, precisionAware=0.5928  TP:38 FP:34 FN:15 TN:41

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 55: auc=0.7479, f1=0.6429, recall=0.6792, precision=0.6102, precisionAware=0.6475, composite=0.6821  TP:36 FP:23 FN:17 TN:52
- epoch 56: auc=0.7404, f1=0.6296, recall=0.6415, precision=0.6182, precisionAware=0.6461, composite=0.6577  TP:34 FP:21 FN:19 TN:54
- epoch 45: auc=0.7321, f1=0.6429, recall=0.6792, precision=0.6102, precisionAware=0.6444, composite=0.6789  TP:36 FP:23 FN:17 TN:52
- epoch 54: auc=0.7459, f1=0.6435, recall=0.6981, precision=0.5968, precisionAware=0.6406, composite=0.6913  TP:37 FP:25 FN:16 TN:50
