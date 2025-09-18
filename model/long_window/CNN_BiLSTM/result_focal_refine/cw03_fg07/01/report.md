# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw03_fg07 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.85 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7002
- F1: 0.4267
- Recall: 0.3048
## Confusion matrix (TP/FP/FN/TN)
- TP: 32
- FP: 13
- FN: 73
- TN: 138

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 45: auc=0.7064, f1=0.6000, recall=0.6226, score=0.6326  TP: 33 FP: 24 FN: 20 TN: 51
- epoch 31: auc=0.6923, f1=0.5926, recall=0.6038, score=0.6181  TP: 32 FP: 23 FN: 21 TN: 52
- epoch 43: auc=0.7203, f1=0.5825, recall=0.5660, score=0.6018  TP: 30 FP: 20 FN: 23 TN: 55
- epoch 36: auc=0.6938, f1=0.5825, recall=0.5660, score=0.5965  TP: 30 FP: 20 FN: 23 TN: 55
