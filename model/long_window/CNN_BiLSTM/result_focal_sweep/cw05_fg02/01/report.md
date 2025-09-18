# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw05_fg02 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.8 --class_weight_pos 1.2 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7199
- F1: 0.3916
- Recall: 0.2667
## Confusion matrix (TP/FP/FN/TN)
- TP: 28
- FP: 10
- FN: 77
- TN: 141

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 41: auc=0.7094, f1=0.5349, recall=0.4340, score=0.5193  TP: 23 FP: 10 FN: 30 TN: 65
- epoch 36: auc=0.6712, f1=0.5106, recall=0.4528, score=0.5138  TP: 24 FP: 17 FN: 29 TN: 58
- epoch 39: auc=0.6830, f1=0.4944, recall=0.4151, score=0.4925  TP: 22 FP: 14 FN: 31 TN: 61
- epoch 40: auc=0.7077, f1=0.4634, recall=0.3585, score=0.4598  TP: 19 FP: 10 FN: 34 TN: 65
