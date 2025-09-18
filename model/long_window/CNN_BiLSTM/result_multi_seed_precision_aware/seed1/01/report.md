# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed1 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 1 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7430
- F1: 0.6273
- Recall: 0.6571
- Precision: 0.6000
- Composite Score: 0.6654 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6368 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 69
- FP: 46
- FN: 36
- TN: 105

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 55: auc=0.7220, f1=0.5806, recall=0.6792, precision=0.5070, score=0.6582, precisionAware=0.5721  TP:36 FP:35 FN:17 TN:40
- epoch 57: auc=0.7545, f1=0.5862, recall=0.6415, precision=0.5397, score=0.6475, precisionAware=0.5966  TP:34 FP:29 FN:19 TN:46
- epoch 53: auc=0.7442, f1=0.5862, recall=0.6415, precision=0.5397, score=0.6454, precisionAware=0.5945  TP:34 FP:29 FN:19 TN:46
- epoch 41: auc=0.6926, f1=0.5738, recall=0.6604, precision=0.5072, score=0.6408, precisionAware=0.5643  TP:35 FP:34 FN:18 TN:41

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 29: auc=0.7220, f1=0.6061, recall=0.5660, precision=0.6522, precisionAware=0.6523, composite=0.6092  TP:30 FP:16 FN:23 TN:59
- epoch 54: auc=0.7668, f1=0.5962, recall=0.5849, precision=0.6078, precisionAware=0.6361, composite=0.6247  TP:31 FP:20 FN:22 TN:55
- epoch 31: auc=0.7283, f1=0.5474, recall=0.4906, precision=0.6190, precisionAware=0.6194, composite=0.5552  TP:26 FP:16 FN:27 TN:59
- epoch 60: auc=0.7494, f1=0.5926, recall=0.6038, precision=0.5818, precisionAware=0.6186, composite=0.6296  TP:32 FP:23 FN:21 TN:52
