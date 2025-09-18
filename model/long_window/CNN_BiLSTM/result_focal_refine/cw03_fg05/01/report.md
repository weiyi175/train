# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw03_fg05 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.85 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7184
- F1: 0.5495
- Recall: 0.4762
## Confusion matrix (TP/FP/FN/TN)
- TP: 50
- FP: 27
- FN: 55
- TN: 124

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 48: auc=0.7112, f1=0.5631, recall=0.5472, score=0.5848  TP: 29 FP: 21 FN: 24 TN: 54
- epoch 46: auc=0.7014, f1=0.5631, recall=0.5472, score=0.5828  TP: 29 FP: 21 FN: 24 TN: 54
- epoch 44: auc=0.6830, f1=0.5545, recall=0.5283, score=0.5671  TP: 28 FP: 20 FN: 25 TN: 55
- epoch 52: auc=0.7258, f1=0.5417, recall=0.4906, score=0.5529  TP: 26 FP: 17 FN: 27 TN: 58
