# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_slipce/seed1 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 1 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7084
- F1: 0.6513
- Recall: 0.6141
- Precision: 0.6933
- Composite Score: 0.6441 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6837 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 113
- FP: 50
- FN: 71
- TN: 120

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 42: auc=0.7308, f1=0.6780, recall=0.7547, precision=0.6154, score=0.7269, precisionAware=0.6572  TP:40 FP:25 FN:13 TN:50
- epoch 60: auc=0.7557, f1=0.6667, recall=0.7170, precision=0.6230, score=0.7096, precisionAware=0.6626  TP:38 FP:23 FN:15 TN:52
- epoch 65: auc=0.7507, f1=0.6333, recall=0.7170, precision=0.5672, score=0.6986, precisionAware=0.6237  TP:38 FP:29 FN:15 TN:46
- epoch 69: auc=0.7439, f1=0.6491, recall=0.6981, precision=0.6066, score=0.6926, precisionAware=0.6468  TP:37 FP:24 FN:16 TN:51

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 70: auc=0.7738, f1=0.6346, recall=0.6226, precision=0.6471, precisionAware=0.6687, composite=0.6565  TP:33 FP:18 FN:20 TN:57
- epoch 60: auc=0.7557, f1=0.6667, recall=0.7170, precision=0.6230, precisionAware=0.6626, composite=0.7096  TP:38 FP:23 FN:15 TN:52
- epoch 58: auc=0.7723, f1=0.6139, recall=0.5849, precision=0.6458, precisionAware=0.6615, composite=0.6311  TP:31 FP:17 FN:22 TN:58
- epoch 66: auc=0.7708, f1=0.6286, recall=0.6226, precision=0.6346, precisionAware=0.6600, composite=0.6541  TP:33 FP:19 FN:20 TN:56
