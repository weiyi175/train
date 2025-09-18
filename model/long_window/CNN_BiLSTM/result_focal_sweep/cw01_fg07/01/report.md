# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw01_fg07 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7363
- F1: 0.6140
- Recall: 0.6286
## Confusion matrix (TP/FP/FN/TN)
- TP: 66
- FP: 44
- FN: 39
- TN: 107

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 58: auc=0.7197, f1=0.6462, recall=0.7925, score=0.7340  TP: 42 FP: 35 FN: 11 TN: 40
- epoch 35: auc=0.6996, f1=0.6557, recall=0.7547, score=0.7140  TP: 40 FP: 29 FN: 13 TN: 46
- epoch 49: auc=0.7233, f1=0.6250, recall=0.7547, score=0.7095  TP: 40 FP: 35 FN: 13 TN: 40
- epoch 56: auc=0.7130, f1=0.6202, recall=0.7547, score=0.7060  TP: 40 FP: 36 FN: 13 TN: 39
