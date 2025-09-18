# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw02_fg03 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7434
- F1: 0.5176
- Recall: 0.4190
## Confusion matrix (TP/FP/FN/TN)
- TP: 44
- FP: 21
- FN: 61
- TN: 130

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 61: auc=0.7882, f1=0.6602, recall=0.6415, score=0.6764  TP: 34 FP: 16 FN: 19 TN: 59
- epoch 47: auc=0.7696, f1=0.6296, recall=0.6415, score=0.6636  TP: 34 FP: 21 FN: 19 TN: 54
- epoch 50: auc=0.7746, f1=0.6346, recall=0.6226, score=0.6566  TP: 33 FP: 18 FN: 20 TN: 57
- epoch 53: auc=0.7852, f1=0.6186, recall=0.5660, score=0.6256  TP: 30 FP: 14 FN: 23 TN: 61
