# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed48 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 48 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6728
- F1: 0.6000
- Recall: 0.5707
- Precision: 0.6325
- Composite Score: 0.5999 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6308 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 105
- FP: 61
- FN: 79
- TN: 109

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.5084, f1=0.5432, recall=0.8302, precision=0.4037, score=0.6797, precisionAware=0.4665  TP:44 FP:65 FN:9 TN:10
- epoch 60: auc=0.6883, f1=0.5455, recall=0.5660, precision=0.5263, score=0.5843, precisionAware=0.5645  TP:30 FP:27 FN:23 TN:48
- epoch 59: auc=0.6855, f1=0.5310, recall=0.5660, precision=0.5000, score=0.5794, precisionAware=0.5464  TP:30 FP:30 FN:23 TN:45
- epoch 54: auc=0.6813, f1=0.5321, recall=0.5472, precision=0.5179, score=0.5695, precisionAware=0.5548  TP:29 FP:27 FN:24 TN:48

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 2: auc=0.5781, f1=0.0727, recall=0.0377, precision=1.0000, precisionAware=0.6374, composite=0.1563  TP:2 FP:0 FN:51 TN:75
- epoch 5: auc=0.6106, f1=0.0370, recall=0.0189, precision=1.0000, precisionAware=0.6332, composite=0.1427  TP:1 FP:0 FN:52 TN:75
- epoch 57: auc=0.6672, f1=0.5437, recall=0.5283, precision=0.5600, precisionAware=0.5765, composite=0.5607  TP:28 FP:22 FN:25 TN:53
- epoch 56: auc=0.6692, f1=0.5385, recall=0.5283, precision=0.5490, precisionAware=0.5699, composite=0.5595  TP:28 FP:23 FN:25 TN:52
