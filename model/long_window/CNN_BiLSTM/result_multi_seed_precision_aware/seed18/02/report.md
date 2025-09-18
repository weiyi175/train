# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed18 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 18 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7137
- F1: 0.5797
- Recall: 0.5714
- Precision: 0.5882
- Composite Score: 0.6024 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6108 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 60
- FP: 42
- FN: 45
- TN: 109

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 59: auc=0.8040, f1=0.6727, recall=0.6981, precision=0.6491, score=0.7117, precisionAware=0.6872  TP:37 FP:20 FN:16 TN:55
- epoch 43: auc=0.7753, f1=0.6372, recall=0.6792, precision=0.6000, score=0.6858, precisionAware=0.6462  TP:36 FP:24 FN:17 TN:51
- epoch 51: auc=0.7814, f1=0.6364, recall=0.6604, precision=0.6140, score=0.6774, precisionAware=0.6542  TP:35 FP:22 FN:18 TN:53
- epoch 38: auc=0.7738, f1=0.6355, recall=0.6415, precision=0.6296, score=0.6662, precisionAware=0.6602  TP:34 FP:20 FN:19 TN:55

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 21: auc=0.7484, f1=0.6452, recall=0.5660, precision=0.7500, precisionAware=0.7182, composite=0.6263  TP:30 FP:10 FN:23 TN:65
- epoch 58: auc=0.7824, f1=0.5814, recall=0.4717, precision=0.7576, precisionAware=0.7097, composite=0.5667  TP:25 FP:8 FN:28 TN:67
- epoch 20: auc=0.7394, f1=0.6304, recall=0.5472, precision=0.7436, precisionAware=0.7088, composite=0.6106  TP:29 FP:10 FN:24 TN:65
- epoch 24: auc=0.7391, f1=0.6237, recall=0.5472, precision=0.7250, precisionAware=0.6974, composite=0.6085  TP:29 FP:11 FN:24 TN:64
