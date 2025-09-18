# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw02_fg09 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6670
- F1: 0.4671
- Recall: 0.3714
## Confusion matrix (TP/FP/FN/TN)
- TP: 39
- FP: 23
- FN: 66
- TN: 128

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 30: auc=0.6533, f1=0.5357, recall=0.5660, score=0.5744  TP: 30 FP: 29 FN: 23 TN: 46
- epoch 23: auc=0.6709, f1=0.5098, recall=0.4906, score=0.5324  TP: 26 FP: 23 FN: 27 TN: 52
- epoch 29: auc=0.6717, f1=0.5102, recall=0.4717, score=0.5232  TP: 25 FP: 20 FN: 28 TN: 55
- epoch 20: auc=0.6642, f1=0.5000, recall=0.4717, score=0.5187  TP: 25 FP: 22 FN: 28 TN: 53
