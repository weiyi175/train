# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed14 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 14 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7512
- F1: 0.6432
- Recall: 0.6952
- Precision: 0.5984
- Composite Score: 0.6908 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6424 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 73
- FP: 49
- FN: 32
- TN: 102

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 55: auc=0.7160, f1=0.6154, recall=0.6792, precision=0.5625, score=0.6674, precisionAware=0.6091  TP:36 FP:28 FN:17 TN:47
- epoch 50: auc=0.7213, f1=0.5766, recall=0.6038, precision=0.5517, score=0.6191, precisionAware=0.5931  TP:32 FP:26 FN:21 TN:49
- epoch 33: auc=0.7019, f1=0.5741, recall=0.5849, precision=0.5636, score=0.6051, precisionAware=0.5944  TP:31 FP:24 FN:22 TN:51
- epoch 28: auc=0.6888, f1=0.5794, recall=0.5849, precision=0.5741, score=0.6040, precisionAware=0.5986  TP:31 FP:23 FN:22 TN:52

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 26: auc=0.7270, f1=0.5714, recall=0.5283, precision=0.6222, precisionAware=0.6279, composite=0.5810  TP:28 FP:17 FN:25 TN:58
- epoch 14: auc=0.6800, f1=0.4557, recall=0.3396, precision=0.6923, precisionAware=0.6189, composite=0.4425  TP:18 FP:8 FN:35 TN:67
- epoch 58: auc=0.7449, f1=0.5217, recall=0.4528, precision=0.6154, precisionAware=0.6132, composite=0.5319  TP:24 FP:15 FN:29 TN:60
- epoch 52: auc=0.7253, f1=0.5510, recall=0.5094, precision=0.6000, precisionAware=0.6104, composite=0.5651  TP:27 FP:18 FN:26 TN:57
