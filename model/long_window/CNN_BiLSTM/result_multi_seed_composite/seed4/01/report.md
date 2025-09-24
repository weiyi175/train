# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite/seed4 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 4 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6681
- F1: 0.6761
- Recall: 0.7826
- Precision: 0.5950
- Composite Score: 0.7277 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6340 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 144
- FP: 98
- FN: 40
- TN: 72

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 37: auc=0.7871, f1=0.7397, recall=0.8571, precision=0.6506, score=0.8079, precisionAware=0.7046  TP:54 FP:29 FN:9 TN:36
- epoch 42: auc=0.7836, f1=0.7397, recall=0.8571, precision=0.6506, score=0.8072, precisionAware=0.7039  TP:54 FP:29 FN:9 TN:36
- epoch 58: auc=0.7714, f1=0.7260, recall=0.8413, precision=0.6386, score=0.7927, precisionAware=0.6914  TP:53 FP:30 FN:10 TN:35
- epoch 45: auc=0.7861, f1=0.7286, recall=0.8095, precision=0.6623, score=0.7805, precisionAware=0.7070  TP:51 FP:26 FN:12 TN:39

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 43: auc=0.7956, f1=0.6542, recall=0.5556, precision=0.7955, precisionAware=0.7531, composite=0.6332  TP:35 FP:9 FN:28 TN:56
- epoch 62: auc=0.7863, f1=0.7344, recall=0.7460, precision=0.7231, precisionAware=0.7391, composite=0.7506  TP:47 FP:18 FN:16 TN:47
- epoch 46: auc=0.7885, f1=0.6607, recall=0.5873, precision=0.7551, precisionAware=0.7335, composite=0.6496  TP:37 FP:12 FN:26 TN:53
- epoch 66: auc=0.8056, f1=0.7368, recall=0.7778, precision=0.7000, precisionAware=0.7322, composite=0.7711  TP:49 FP:21 FN:14 TN:44
