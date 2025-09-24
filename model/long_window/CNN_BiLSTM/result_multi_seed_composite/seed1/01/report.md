# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce_thresh040/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_multi_seed_composite/seed1 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 1 --checkpoint_metric composite --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6823
- F1: 0.6098
- Recall: 0.5435
- Precision: 0.6944
- Composite Score: 0.5911 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6666 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 100
- FP: 44
- FN: 84
- TN: 126

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 69: auc=0.8081, f1=0.7815, recall=0.9365, precision=0.6705, score=0.8643, precisionAware=0.7313  TP:59 FP:29 FN:4 TN:36
- epoch 48: auc=0.8059, f1=0.7815, recall=0.9365, precision=0.6705, score=0.8639, precisionAware=0.7308  TP:59 FP:29 FN:4 TN:36
- epoch 42: auc=0.8066, f1=0.7733, recall=0.9206, precision=0.6667, score=0.8536, precisionAware=0.7267  TP:58 FP:29 FN:5 TN:36
- epoch 59: auc=0.8078, f1=0.7917, recall=0.9048, precision=0.7037, score=0.8514, precisionAware=0.7509  TP:57 FP:24 FN:6 TN:41

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 31: auc=0.7990, f1=0.7350, recall=0.6825, precision=0.7963, precisionAware=0.7785, composite=0.7216  TP:43 FP:11 FN:20 TN:54
- epoch 70: auc=0.8259, f1=0.6964, recall=0.6190, precision=0.7959, precisionAware=0.7721, composite=0.6836  TP:39 FP:10 FN:24 TN:55
- epoch 49: auc=0.8249, f1=0.7333, recall=0.6984, precision=0.7719, precisionAware=0.7709, composite=0.7342  TP:44 FP:13 FN:19 TN:52
- epoch 58: auc=0.7961, f1=0.7656, recall=0.7778, precision=0.7538, precisionAware=0.7658, composite=0.7778  TP:49 FP:16 FN:14 TN:49
