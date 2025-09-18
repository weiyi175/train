# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw06_fg05 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.1 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7326
- F1: 0.5207
- Recall: 0.4190
## Confusion matrix (TP/FP/FN/TN)
- TP: 44
- FP: 20
- FN: 61
- TN: 131

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 66: auc=0.7781, f1=0.6545, recall=0.6792, score=0.6916  TP: 36 FP: 21 FN: 17 TN: 54
- epoch 64: auc=0.7877, f1=0.6296, recall=0.6415, score=0.6672  TP: 34 FP: 21 FN: 19 TN: 54
- epoch 61: auc=0.7675, f1=0.5905, recall=0.5849, score=0.6231  TP: 31 FP: 21 FN: 22 TN: 54
- epoch 59: auc=0.7746, f1=0.5882, recall=0.5660, score=0.6144  TP: 30 FP: 19 FN: 23 TN: 56
