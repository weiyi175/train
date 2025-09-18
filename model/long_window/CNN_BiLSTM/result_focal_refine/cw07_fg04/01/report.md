# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw07_fg04 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.9 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7442
- F1: 0.5089
- Recall: 0.4095
## Confusion matrix (TP/FP/FN/TN)
- TP: 43
- FP: 21
- FN: 62
- TN: 130

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 58: auc=0.7414, f1=0.6476, recall=0.6415, score=0.6633  TP: 34 FP: 18 FN: 19 TN: 57
- epoch 61: auc=0.7525, f1=0.6346, recall=0.6226, score=0.6522  TP: 33 FP: 18 FN: 20 TN: 57
- epoch 59: auc=0.7721, f1=0.6275, recall=0.6038, score=0.6445  TP: 32 FP: 17 FN: 21 TN: 58
- epoch 69: auc=0.7462, f1=0.6095, recall=0.6038, score=0.6340  TP: 32 FP: 20 FN: 21 TN: 55
