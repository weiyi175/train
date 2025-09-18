# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw06_fg04 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.1 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6025
- F1: 0.1008
- Recall: 0.0571
## Confusion matrix (TP/FP/FN/TN)
- TP: 6
- FP: 8
- FN: 99
- TN: 143

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 1: auc=0.5839, f1=0.0984, recall=0.0566, score=0.1746  TP: 3 FP: 5 FN: 50 TN: 70
- epoch 11: auc=0.5839, f1=0.0984, recall=0.0566, score=0.1746  TP: 3 FP: 5 FN: 50 TN: 70
- epoch 3: auc=0.5718, f1=0.0000, recall=0.0000, score=0.1144  TP: 0 FP: 0 FN: 53 TN: 75
- epoch 4: auc=0.5706, f1=0.0000, recall=0.0000, score=0.1141  TP: 0 FP: 0 FN: 53 TN: 75
