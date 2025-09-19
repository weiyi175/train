# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed15 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 15 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7048
- F1: 0.6503
- Recall: 0.6467
- Precision: 0.6538
- Composite Score: 0.6594 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6630 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 119
- FP: 63
- FN: 65
- TN: 107

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 50: auc=0.6951, f1=0.6218, recall=0.6981, precision=0.5606, score=0.6746, precisionAware=0.6059  TP:37 FP:29 FN:16 TN:46
- epoch 44: auc=0.6853, f1=0.6140, recall=0.6604, precision=0.5738, score=0.6515, precisionAware=0.6082  TP:35 FP:26 FN:18 TN:49
- epoch 45: auc=0.6858, f1=0.6087, recall=0.6604, precision=0.5645, score=0.6500, precisionAware=0.6020  TP:35 FP:27 FN:18 TN:48
- epoch 51: auc=0.7011, f1=0.6071, recall=0.6415, precision=0.5763, score=0.6431, precisionAware=0.6105  TP:34 FP:25 FN:19 TN:50

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 4: auc=0.6035, f1=0.0727, recall=0.0377, precision=1.0000, precisionAware=0.6425, composite=0.1614  TP:2 FP:0 FN:51 TN:75
- epoch 3: auc=0.5952, f1=0.0370, recall=0.0189, precision=1.0000, precisionAware=0.6302, composite=0.1396  TP:1 FP:0 FN:52 TN:75
- epoch 57: auc=0.7185, f1=0.5962, recall=0.5849, precision=0.6078, precisionAware=0.6265, composite=0.6150  TP:31 FP:20 FN:22 TN:55
- epoch 55: auc=0.7069, f1=0.6111, recall=0.6226, precision=0.6000, precisionAware=0.6247, composite=0.6360  TP:33 FP:22 FN:20 TN:53
