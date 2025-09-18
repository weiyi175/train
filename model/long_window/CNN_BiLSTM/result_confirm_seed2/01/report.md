# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_confirm_seed2 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --no_early_stop --run_seed 2 --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7391
- F1: 0.6049
- Recall: 0.5905
## Confusion matrix (TP/FP/FN/TN)
- TP: 62
- FP: 38
- FN: 43
- TN: 113

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 53: auc=0.7537, f1=0.6218, recall=0.6981, score=0.6864  TP: 37 FP: 29 FN: 16 TN: 46
- epoch 41: auc=0.7172, f1=0.5938, recall=0.7170, score=0.6801  TP: 38 FP: 37 FN: 15 TN: 38
- epoch 46: auc=0.7431, f1=0.6154, recall=0.6792, score=0.6729  TP: 36 FP: 28 FN: 17 TN: 47
- epoch 44: auc=0.7638, f1=0.6306, recall=0.6604, score=0.6721  TP: 35 FP: 23 FN: 18 TN: 52
