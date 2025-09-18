# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw08_fg08 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7326
- F1: 0.5574
- Recall: 0.4857
## Confusion matrix (TP/FP/FN/TN)
- TP: 51
- FP: 27
- FN: 54
- TN: 124

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 34: auc=0.7099, f1=0.5962, recall=0.5849, score=0.6133  TP: 31 FP: 20 FN: 22 TN: 55
- epoch 30: auc=0.7052, f1=0.5825, recall=0.5660, score=0.5988  TP: 30 FP: 20 FN: 23 TN: 55
- epoch 32: auc=0.7009, f1=0.5773, recall=0.5283, score=0.5775  TP: 28 FP: 16 FN: 25 TN: 59
- epoch 27: auc=0.7145, f1=0.5745, recall=0.5094, score=0.5700  TP: 27 FP: 14 FN: 26 TN: 61
