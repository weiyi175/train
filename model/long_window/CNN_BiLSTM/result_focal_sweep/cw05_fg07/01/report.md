# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw05_fg07 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.8 --class_weight_pos 1.2 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7365
- F1: 0.5979
- Recall: 0.5524
## Confusion matrix (TP/FP/FN/TN)
- TP: 58
- FP: 31
- FN: 47
- TN: 120

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 60: auc=0.7245, f1=0.6406, recall=0.7736, score=0.7239  TP: 41 FP: 34 FN: 12 TN: 41
- epoch 58: auc=0.7079, f1=0.6176, recall=0.7925, score=0.7231  TP: 42 FP: 41 FN: 11 TN: 34
- epoch 41: auc=0.7132, f1=0.6308, recall=0.7736, score=0.7187  TP: 41 FP: 36 FN: 12 TN: 39
- epoch 34: auc=0.6999, f1=0.6212, recall=0.7736, score=0.7131  TP: 41 FP: 38 FN: 12 TN: 37
