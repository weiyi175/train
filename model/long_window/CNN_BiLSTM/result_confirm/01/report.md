# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_confirm --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --no_early_stop --run_seed None --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7306
- F1: 0.6154
- Recall: 0.6476
## Confusion matrix (TP/FP/FN/TN)
- TP: 68
- FP: 48
- FN: 37
- TN: 103

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 34: auc=0.7117, f1=0.6393, recall=0.7358, score=0.7021  TP: 39 FP: 30 FN: 14 TN: 45
- epoch 47: auc=0.7077, f1=0.6094, recall=0.7358, score=0.6923  TP: 39 FP: 36 FN: 14 TN: 39
- epoch 65: auc=0.7494, f1=0.6167, recall=0.6981, score=0.6839  TP: 37 FP: 30 FN: 16 TN: 45
- epoch 32: auc=0.7326, f1=0.6545, recall=0.6792, score=0.6825  TP: 36 FP: 21 FN: 17 TN: 54
