# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw01_fg04 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.75 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7239
- F1: 0.3284
- Recall: 0.2095
## Confusion matrix (TP/FP/FN/TN)
- TP: 22
- FP: 7
- FN: 83
- TN: 144

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 38: auc=0.6511, f1=0.4176, recall=0.3585, score=0.4347  TP: 19 FP: 19 FN: 34 TN: 56
- epoch 39: auc=0.6692, f1=0.4235, recall=0.3396, score=0.4307  TP: 18 FP: 14 FN: 35 TN: 61
- epoch 33: auc=0.6531, f1=0.4096, recall=0.3208, score=0.4139  TP: 17 FP: 13 FN: 36 TN: 62
- epoch 36: auc=0.6787, f1=0.3846, recall=0.2830, score=0.3926  TP: 15 FP: 10 FN: 38 TN: 65
