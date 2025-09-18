# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw05_fg05 --focal_alpha 0.1 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.8 --class_weight_pos 1.2 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7161
- F1: 0.2946
- Recall: 0.1810
## Confusion matrix (TP/FP/FN/TN)
- TP: 19
- FP: 5
- FN: 86
- TN: 146

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 50: auc=0.6546, f1=0.3636, recall=0.2642, score=0.3721  TP: 14 FP: 10 FN: 39 TN: 65
- epoch 46: auc=0.6803, f1=0.3333, recall=0.2264, score=0.3493  TP: 12 FP: 7 FN: 41 TN: 68
- epoch 45: auc=0.6893, f1=0.3099, recall=0.2075, score=0.3346  TP: 11 FP: 7 FN: 42 TN: 68
- epoch 48: auc=0.6850, f1=0.2857, recall=0.1887, score=0.3171  TP: 10 FP: 7 FN: 43 TN: 68
