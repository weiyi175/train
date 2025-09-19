# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed1 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 1 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7210
- F1: 0.6235
- Recall: 0.5489
- Precision: 0.7214
- Composite Score: 0.6057 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6920 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 101
- FP: 39
- FN: 83
- TN: 131

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 43: auc=0.7600, f1=0.6337, recall=0.6038, precision=0.6667, score=0.6440, precisionAware=0.6754  TP:32 FP:16 FN:21 TN:59
- epoch 32: auc=0.7293, f1=0.6275, recall=0.6038, precision=0.6531, score=0.6360, precisionAware=0.6606  TP:32 FP:17 FN:21 TN:58
- epoch 55: auc=0.7701, f1=0.6078, recall=0.5849, precision=0.6327, score=0.6288, precisionAware=0.6527  TP:31 FP:18 FN:22 TN:57
- epoch 45: auc=0.7512, f1=0.6078, recall=0.5849, precision=0.6327, score=0.6250, precisionAware=0.6489  TP:31 FP:18 FN:22 TN:57

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 40: auc=0.7638, f1=0.6170, recall=0.5472, precision=0.7073, precisionAware=0.6915, composite=0.6114  TP:29 FP:12 FN:24 TN:63
- epoch 56: auc=0.7947, f1=0.6022, recall=0.5283, precision=0.7000, precisionAware=0.6896, composite=0.6037  TP:28 FP:12 FN:25 TN:63
- epoch 23: auc=0.7386, f1=0.5934, recall=0.5094, precision=0.7105, precisionAware=0.6810, composite=0.5805  TP:27 FP:11 FN:26 TN:64
- epoch 43: auc=0.7600, f1=0.6337, recall=0.6038, precision=0.6667, precisionAware=0.6754, composite=0.6440  TP:32 FP:16 FN:21 TN:59
