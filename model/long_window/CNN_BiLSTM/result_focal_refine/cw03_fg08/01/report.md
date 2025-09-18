# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw03_fg08 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.85 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7162
- F1: 0.5030
- Recall: 0.4000
## Confusion matrix (TP/FP/FN/TN)
- TP: 42
- FP: 20
- FN: 63
- TN: 131

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 29: auc=0.7323, f1=0.6538, recall=0.6415, score=0.6634  TP: 34 FP: 17 FN: 19 TN: 58
- epoch 32: auc=0.7072, f1=0.6061, recall=0.5660, score=0.6063  TP: 30 FP: 16 FN: 23 TN: 59
- epoch 35: auc=0.7006, f1=0.5741, recall=0.5849, score=0.6048  TP: 31 FP: 24 FN: 22 TN: 51
- epoch 33: auc=0.7077, f1=0.5859, recall=0.5472, score=0.5909  TP: 29 FP: 17 FN: 24 TN: 58
