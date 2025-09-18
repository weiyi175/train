# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw04_fg07 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.95 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7474
- F1: 0.5668
- Recall: 0.5048
## Confusion matrix (TP/FP/FN/TN)
- TP: 53
- FP: 29
- FN: 52
- TN: 122

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 41: auc=0.7190, f1=0.5849, recall=0.5849, score=0.6117  TP: 31 FP: 22 FN: 22 TN: 53
- epoch 38: auc=0.7044, f1=0.5688, recall=0.5849, score=0.6040  TP: 31 FP: 25 FN: 22 TN: 50
- epoch 34: auc=0.7117, f1=0.5825, recall=0.5660, score=0.6001  TP: 30 FP: 20 FN: 23 TN: 55
- epoch 31: auc=0.7112, f1=0.5918, recall=0.5472, score=0.5934  TP: 29 FP: 16 FN: 24 TN: 59
