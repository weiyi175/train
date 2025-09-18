# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_confirm --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --no_early_stop --run_seed None --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7246
- F1: 0.6201
- Recall: 0.6762
## Confusion matrix (TP/FP/FN/TN)
- TP: 71
- FP: 53
- FN: 34
- TN: 98

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 65: auc=0.6933, f1=0.6015, recall=0.7547, score=0.6965  TP: 40 FP: 40 FN: 13 TN: 35
- epoch 54: auc=0.7293, f1=0.6207, recall=0.6792, score=0.6717  TP: 36 FP: 27 FN: 17 TN: 48
- epoch 35: auc=0.6762, f1=0.5846, recall=0.7170, score=0.6691  TP: 38 FP: 39 FN: 15 TN: 36
- epoch 50: auc=0.7089, f1=0.5873, recall=0.6981, score=0.6670  TP: 37 FP: 36 FN: 16 TN: 39
