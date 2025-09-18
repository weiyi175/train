# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw04_fg03 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.95 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6857
- F1: 0.5528
- Recall: 0.5238
## Confusion matrix (TP/FP/FN/TN)
- TP: 55
- FP: 39
- FN: 50
- TN: 112

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 61: auc=0.7275, f1=0.6316, recall=0.6792, score=0.6746  TP: 36 FP: 25 FN: 17 TN: 50
- epoch 60: auc=0.7374, f1=0.6214, recall=0.6038, score=0.6358  TP: 32 FP: 18 FN: 21 TN: 57
- epoch 64: auc=0.7140, f1=0.5905, recall=0.5849, score=0.6124  TP: 31 FP: 21 FN: 22 TN: 54
- epoch 58: auc=0.7301, f1=0.5941, recall=0.5660, score=0.6072  TP: 30 FP: 18 FN: 23 TN: 57
