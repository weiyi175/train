# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_center/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite_center/seed5 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 5 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6482
- F1: 0.6635
- Recall: 0.7609
- Precision: 0.5882
- Composite Score: 0.7091 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6228 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 140
- FP: 98
- FN: 44
- TN: 72

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 47: auc=0.7257, f1=0.6715, recall=0.8519, precision=0.5542, score=0.7725, precisionAware=0.6237  TP:46 FP:37 FN:8 TN:37
- epoch 50: auc=0.7347, f1=0.6519, recall=0.8148, precision=0.5432, score=0.7499, precisionAware=0.6141  TP:44 FP:37 FN:10 TN:37
- epoch 67: auc=0.7528, f1=0.6774, recall=0.7778, precision=0.6000, score=0.7427, precisionAware=0.6538  TP:42 FP:28 FN:12 TN:46
- epoch 60: auc=0.7305, f1=0.6466, recall=0.7963, precision=0.5443, score=0.7382, precisionAware=0.6122  TP:43 FP:36 FN:11 TN:38

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 28: auc=0.7370, f1=0.6604, recall=0.6481, precision=0.6731, precisionAware=0.6820, composite=0.6696  TP:35 FP:17 FN:19 TN:57
- epoch 27: auc=0.7402, f1=0.5870, recall=0.5000, precision=0.7105, precisionAware=0.6794, composite=0.5741  TP:27 FP:11 FN:27 TN:63
- epoch 33: auc=0.7518, f1=0.5618, recall=0.4630, precision=0.7143, precisionAware=0.6760, composite=0.5504  TP:25 FP:10 FN:29 TN:64
- epoch 24: auc=0.7310, f1=0.5979, recall=0.5370, precision=0.6744, precisionAware=0.6628, composite=0.5941  TP:29 FP:14 FN:25 TN:60
