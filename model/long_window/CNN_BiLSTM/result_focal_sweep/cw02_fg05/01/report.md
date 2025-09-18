# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw02_fg05 --focal_alpha 0.1 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6957
- F1: 0.5588
- Recall: 0.5429
## Confusion matrix (TP/FP/FN/TN)
- TP: 57
- FP: 42
- FN: 48
- TN: 109

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 66: auc=0.6896, f1=0.5421, recall=0.5472, score=0.5741  TP: 29 FP: 25 FN: 24 TN: 50
- epoch 63: auc=0.7062, f1=0.5306, recall=0.4906, score=0.5457  TP: 26 FP: 19 FN: 27 TN: 56
- epoch 70: auc=0.6730, f1=0.4954, recall=0.5094, score=0.5379  TP: 27 FP: 29 FN: 26 TN: 46
- epoch 56: auc=0.7177, f1=0.5263, recall=0.4717, score=0.5373  TP: 25 FP: 17 FN: 28 TN: 58
