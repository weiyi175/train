# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw07_fg02 --focal_alpha 0.22 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.9 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6663
- F1: 0.4671
- Recall: 0.3714
## Confusion matrix (TP/FP/FN/TN)
- TP: 39
- FP: 23
- FN: 66
- TN: 128

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 44: auc=0.6848, f1=0.5818, recall=0.6038, score=0.6134  TP: 32 FP: 25 FN: 21 TN: 50
- epoch 41: auc=0.7182, f1=0.5688, recall=0.5849, score=0.6067  TP: 31 FP: 25 FN: 22 TN: 50
- epoch 37: auc=0.7052, f1=0.5825, recall=0.5660, score=0.5988  TP: 30 FP: 20 FN: 23 TN: 55
- epoch 47: auc=0.7243, f1=0.5524, recall=0.5472, score=0.5842  TP: 29 FP: 23 FN: 24 TN: 52
