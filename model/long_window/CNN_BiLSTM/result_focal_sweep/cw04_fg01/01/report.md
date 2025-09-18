# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw04_fg01 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.2 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7329
- F1: 0.5816
- Recall: 0.5429
## Confusion matrix (TP/FP/FN/TN)
- TP: 57
- FP: 34
- FN: 48
- TN: 117

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 64: auc=0.7791, f1=0.6729, recall=0.6792, score=0.6973  TP: 36 FP: 18 FN: 17 TN: 57
- epoch 67: auc=0.7743, f1=0.6667, recall=0.6604, score=0.6851  TP: 35 FP: 17 FN: 18 TN: 58
- epoch 53: auc=0.7630, f1=0.6415, recall=0.6415, score=0.6658  TP: 34 FP: 19 FN: 19 TN: 56
- epoch 36: auc=0.7185, f1=0.6250, recall=0.6604, score=0.6614  TP: 35 FP: 24 FN: 18 TN: 51
