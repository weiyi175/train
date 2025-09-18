# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw03_fg04 --focal_alpha 0.2 --focal_gamma_start 0.0 --focal_gamma_end 2.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.6 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7268
- F1: 0.4706
- Recall: 0.3810
## Confusion matrix (TP/FP/FN/TN)
- TP: 40
- FP: 25
- FN: 65
- TN: 126

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 66: auc=0.7907, f1=0.6379, recall=0.6981, score=0.6986  TP: 37 FP: 26 FN: 16 TN: 49
- epoch 68: auc=0.8141, f1=0.6481, recall=0.6604, score=0.6875  TP: 35 FP: 20 FN: 18 TN: 55
- epoch 60: auc=0.8043, f1=0.6481, recall=0.6604, score=0.6855  TP: 35 FP: 20 FN: 18 TN: 55
- epoch 64: auc=0.7975, f1=0.6481, recall=0.6604, score=0.6841  TP: 35 FP: 20 FN: 18 TN: 55
