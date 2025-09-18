# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed7 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 7 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7280
- F1: 0.6301
- Recall: 0.6571
- Precision: 0.6053
- Composite Score: 0.6632 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6373 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 69
- FP: 45
- FN: 36
- TN: 106

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 59: auc=0.7794, f1=0.6306, recall=0.6604, precision=0.6034, score=0.6753, precisionAware=0.6468  TP:35 FP:23 FN:18 TN:52
- epoch 46: auc=0.7509, f1=0.6250, recall=0.6604, precision=0.5932, score=0.6679, precisionAware=0.6343  TP:35 FP:24 FN:18 TN:51
- epoch 60: auc=0.7796, f1=0.6355, recall=0.6415, precision=0.6296, score=0.6673, precisionAware=0.6614  TP:34 FP:20 FN:19 TN:55
- epoch 38: auc=0.7633, f1=0.6471, recall=0.6226, precision=0.6735, score=0.6581, precisionAware=0.6835  TP:33 FP:16 FN:20 TN:59

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 45: auc=0.7743, f1=0.6237, recall=0.5472, precision=0.7250, precisionAware=0.7045, composite=0.6155  TP:29 FP:11 FN:24 TN:64
- epoch 38: auc=0.7633, f1=0.6471, recall=0.6226, precision=0.6735, precisionAware=0.6835, composite=0.6581  TP:33 FP:16 FN:20 TN:59
- epoch 29: auc=0.7454, f1=0.6337, recall=0.6038, precision=0.6667, precisionAware=0.6725, composite=0.6411  TP:32 FP:16 FN:21 TN:59
- epoch 40: auc=0.7658, f1=0.5714, recall=0.4906, precision=0.6842, precisionAware=0.6667, composite=0.5699  TP:26 FP:12 FN:27 TN:63
