# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw06_fg07 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.1 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6269
- F1: 0.1217
- Recall: 0.0667
## Confusion matrix (TP/FP/FN/TN)
- TP: 7
- FP: 3
- FN: 98
- TN: 148

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 10: auc=0.5922, f1=0.2258, recall=0.1321, score=0.2522  TP: 7 FP: 2 FN: 46 TN: 73
- epoch 9: auc=0.5912, f1=0.2000, recall=0.1132, score=0.2348  TP: 6 FP: 1 FN: 47 TN: 74
- epoch 7: auc=0.5955, f1=0.1724, recall=0.0943, score=0.2180  TP: 5 FP: 0 FN: 48 TN: 75
- epoch 8: auc=0.5945, f1=0.1724, recall=0.0943, score=0.2178  TP: 5 FP: 0 FN: 48 TN: 75
