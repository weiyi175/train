# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --val_windows None --test_windows /home/user/projects/train/test_data/slipce/windows_npz.npz --epochs 60 --batch 64 --accumulate_steps 8 --result_dir result_ensemble_weights/seed12 --focal_alpha 0.5 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed 12 --checkpoint_metric precision_aware --class_weight_neg 1.05 --class_weight_pos 1.0 --mask_threshold 0.7 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6947
- F1: 0.6700
- Recall: 0.7337
- Precision: 0.6164
- Composite Score: 0.7068 (0.5*Recall + 0.3*F1 + 0.2*AUC)
- Precision-aware Score: 0.6482 (0.5*Precision + 0.3*F1 + 0.2*AUC)
## Confusion matrix (TP/FP/FN/TN)
- TP: 135
- FP: 84
- FN: 49
- TN: 86

## Top 4 epochs by Composite (0.5*Recall + 0.3*F1 + 0.2*AUC)
- epoch 47: auc=0.7361, f1=0.6667, recall=0.8679, precision=0.5412, score=0.7812, precisionAware=0.6178  TP:46 FP:39 FN:7 TN:36
- epoch 49: auc=0.7293, f1=0.6619, recall=0.8679, precision=0.5349, score=0.7784, precisionAware=0.6119  TP:46 FP:40 FN:7 TN:35
- epoch 48: auc=0.7270, f1=0.6309, recall=0.8868, precision=0.4896, score=0.7781, precisionAware=0.5795  TP:47 FP:49 FN:6 TN:26
- epoch 55: auc=0.7311, f1=0.6617, recall=0.8302, precision=0.5500, score=0.7598, precisionAware=0.6197  TP:44 FP:36 FN:9 TN:39

## Top 4 epochs by Precision-aware (0.5*Precision + 0.3*F1 + 0.2*AUC)
- epoch 1: auc=0.6088, f1=0.0370, recall=0.0189, precision=1.0000, precisionAware=0.6329, composite=0.1423  TP:1 FP:0 FN:52 TN:75
- epoch 51: auc=0.7497, f1=0.6446, recall=0.7358, precision=0.5735, precisionAware=0.6301, composite=0.7113  TP:39 FP:29 FN:14 TN:46
- epoch 59: auc=0.7306, f1=0.6614, recall=0.7925, precision=0.5676, precisionAware=0.6283, composite=0.7408  TP:42 FP:32 FN:11 TN:43
- epoch 46: auc=0.7384, f1=0.6615, recall=0.8113, precision=0.5584, precisionAware=0.6254, composite=0.7518  TP:43 FP:34 FN:10 TN:41
