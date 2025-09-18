# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw07_fg06 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.9 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7483
- F1: 0.5311
- Recall: 0.4476
## Confusion matrix (TP/FP/FN/TN)
- TP: 47
- FP: 25
- FN: 58
- TN: 126

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 47: auc=0.7366, f1=0.5849, recall=0.5849, score=0.6152  TP: 31 FP: 22 FN: 22 TN: 53
- epoch 50: auc=0.7371, f1=0.5743, recall=0.5472, score=0.5933  TP: 29 FP: 19 FN: 24 TN: 56
- epoch 42: auc=0.7142, f1=0.5347, recall=0.5094, score=0.5580  TP: 27 FP: 21 FN: 26 TN: 54
- epoch 52: auc=0.7215, f1=0.5200, recall=0.4906, score=0.5456  TP: 26 FP: 21 FN: 27 TN: 54
