# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed47 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 47 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6893
- F1: 0.6600
- Recall: 0.7228
- Precision: 0.6073
- Composite Score: 0.6973 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6395 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 133
- FP: 86
- FN: 51
- TN: 84

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 60: auc=0.7379, f1=0.7068, recall=0.8868, precision=0.5875, score=0.8030, precisionAware=0.6534  TP:47 FP:33 FN:6 TN:42
- epoch 58: auc=0.7369, f1=0.6977, recall=0.8491, precision=0.5921, score=0.7812, precisionAware=0.6527  TP:45 FP:31 FN:8 TN:44
- epoch 59: auc=0.7421, f1=0.7097, recall=0.8302, precision=0.6197, score=0.7764, precisionAware=0.6712  TP:44 FP:27 FN:9 TN:48
- epoch 51: auc=0.7358, f1=0.6716, recall=0.8491, precision=0.5556, score=0.7732, precisionAware=0.6264  TP:45 FP:36 FN:8 TN:39

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 59: auc=0.7421, f1=0.7097, recall=0.8302, precision=0.6197, precisionAware=0.6712, composite=0.7764  TP:44 FP:27 FN:9 TN:48
- epoch 55: auc=0.7391, f1=0.6885, recall=0.7925, precision=0.6087, precisionAware=0.6587, composite=0.7506  TP:42 FP:27 FN:11 TN:48
- epoch 60: auc=0.7379, f1=0.7068, recall=0.8868, precision=0.5875, precisionAware=0.6534, composite=0.8030  TP:47 FP:33 FN:6 TN:42
- epoch 58: auc=0.7369, f1=0.6977, recall=0.8491, precision=0.5921, precisionAware=0.6527, composite=0.7812  TP:45 FP:31 FN:8 TN:44
