# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_precision_aware/seed9 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 9 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6824
- F1: 0.5288
- Recall: 0.5238
- Precision: 0.5340
- Composite Score: 0.5570 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.5621 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 55
- FP: 48
- FN: 50
- TN: 103

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 33: auc=0.7137, f1=0.6032, recall=0.7170, precision=0.5205, score=0.6822, precisionAware=0.5840  TP:38 FP:35 FN:15 TN:40
- epoch 54: auc=0.7011, f1=0.6218, recall=0.6981, precision=0.5606, score=0.6758, precisionAware=0.6071  TP:37 FP:29 FN:16 TN:46
- epoch 49: auc=0.7263, f1=0.6261, recall=0.6792, precision=0.5806, score=0.6727, precisionAware=0.6234  TP:36 FP:26 FN:17 TN:49
- epoch 59: auc=0.7301, f1=0.6102, recall=0.6792, precision=0.5538, score=0.6687, precisionAware=0.6060  TP:36 FP:29 FN:17 TN:46

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 53: auc=0.7542, f1=0.6226, recall=0.6226, precision=0.6226, precisionAware=0.6490, composite=0.6490  TP:33 FP:20 FN:20 TN:55
- epoch 29: auc=0.7439, f1=0.5918, recall=0.5472, precision=0.6444, precisionAware=0.6486, composite=0.5999  TP:29 FP:16 FN:24 TN:59
- epoch 43: auc=0.7469, f1=0.6000, recall=0.5660, precision=0.6383, precisionAware=0.6485, composite=0.6124  TP:30 FP:17 FN:23 TN:58
- epoch 51: auc=0.7434, f1=0.6226, recall=0.6226, precision=0.6226, precisionAware=0.6468, composite=0.6468  TP:33 FP:20 FN:20 TN:55
