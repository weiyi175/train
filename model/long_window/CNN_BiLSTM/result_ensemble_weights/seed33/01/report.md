# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed33 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 33 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7202
- F1: 0.6457
- Recall: 0.6141
- Precision: 0.6807
- Composite Score: 0.6448 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6781 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 113
- FP: 53
- FN: 71
- TN: 117

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 52: auc=0.7009, f1=0.6345, recall=0.8679, precision=0.5000, score=0.7645, precisionAware=0.5805  TP:46 FP:46 FN:7 TN:29
- epoch 53: auc=0.7047, f1=0.6187, recall=0.8113, precision=0.5000, score=0.7322, precisionAware=0.5765  TP:43 FP:43 FN:10 TN:32
- epoch 60: auc=0.7394, f1=0.6842, recall=0.7358, precision=0.6393, score=0.7211, precisionAware=0.6728  TP:39 FP:22 FN:14 TN:53
- epoch 59: auc=0.7248, f1=0.6612, recall=0.7547, precision=0.5882, score=0.7207, precisionAware=0.6374  TP:40 FP:28 FN:13 TN:47

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7394, f1=0.6842, recall=0.7358, precision=0.6393, precisionAware=0.6728, composite=0.7211  TP:39 FP:22 FN:14 TN:53
- epoch 54: auc=0.7358, f1=0.6609, recall=0.7170, precision=0.6129, precisionAware=0.6519, composite=0.7039  TP:38 FP:24 FN:15 TN:51
- epoch 55: auc=0.7351, f1=0.6306, recall=0.6604, precision=0.6034, precisionAware=0.6379, composite=0.6664  TP:35 FP:23 FN:18 TN:52
- epoch 59: auc=0.7248, f1=0.6612, recall=0.7547, precision=0.5882, precisionAware=0.6374, composite=0.7207  TP:40 FP:28 FN:13 TN:47
