# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw05_fg02 --focal_alpha 0.22 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7227
- F1: 0.4238
- Recall: 0.3048
## Confusion matrix (TP/FP/FN/TN)
- TP: 32
- FP: 14
- FN: 73
- TN: 137

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 37: auc=0.7238, f1=0.5743, recall=0.5472, score=0.5906  TP: 29 FP: 19 FN: 24 TN: 56
- epoch 43: auc=0.7084, f1=0.5437, recall=0.5283, score=0.5689  TP: 28 FP: 22 FN: 25 TN: 53
- epoch 36: auc=0.7087, f1=0.5455, recall=0.5094, score=0.5601  TP: 27 FP: 19 FN: 26 TN: 56
- epoch 41: auc=0.7177, f1=0.5253, recall=0.4906, score=0.5464  TP: 26 FP: 20 FN: 27 TN: 55
