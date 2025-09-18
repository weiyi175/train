# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_confirm_seed3 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --no_early_stop --run_seed 3 --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7505
- F1: 0.6009
- Recall: 0.6667
## Confusion matrix (TP/FP/FN/TN)
- TP: 70
- FP: 58
- FN: 35
- TN: 93

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 56: auc=0.7210, f1=0.6723, recall=0.7547, score=0.7232  TP: 40 FP: 26 FN: 13 TN: 49
- epoch 69: auc=0.7283, f1=0.6667, recall=0.7547, score=0.7230  TP: 40 FP: 27 FN: 13 TN: 48
- epoch 54: auc=0.7253, f1=0.6903, recall=0.7358, score=0.7201  TP: 39 FP: 21 FN: 14 TN: 54
- epoch 51: auc=0.6961, f1=0.6400, recall=0.7547, score=0.7086  TP: 40 FP: 32 FN: 13 TN: 43
