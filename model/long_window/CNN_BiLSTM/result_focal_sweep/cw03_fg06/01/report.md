# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw03_fg06 --focal_alpha 0.1 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.6 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7317
- F1: 0.3453
- Recall: 0.2286
## Confusion matrix (TP/FP/FN/TN)
- TP: 24
- FP: 10
- FN: 81
- TN: 141

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 65: auc=0.7208, f1=0.4250, recall=0.3208, score=0.4320  TP: 17 FP: 10 FN: 36 TN: 65
- epoch 61: auc=0.7517, f1=0.4156, recall=0.3019, score=0.4260  TP: 16 FP: 8 FN: 37 TN: 67
- epoch 64: auc=0.7044, f1=0.4103, recall=0.3019, score=0.4149  TP: 16 FP: 9 FN: 37 TN: 66
- epoch 54: auc=0.7514, f1=0.3562, recall=0.2453, score=0.3798  TP: 13 FP: 7 FN: 40 TN: 68
