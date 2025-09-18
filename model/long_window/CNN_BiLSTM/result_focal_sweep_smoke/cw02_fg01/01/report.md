# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 10 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep_smoke/cw02_fg01 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6531
- F1: 0.1043
- Recall: 0.0571
## Confusion matrix (TP/FP/FN/TN)
- TP: 6
- FP: 4
- FN: 99
- TN: 147

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 10: auc=0.5789, f1=0.0727, recall=0.0377, score=0.1565  TP: 2 FP: 0 FN: 51 TN: 75
- epoch 9: auc=0.5743, f1=0.0370, recall=0.0189, score=0.1354  TP: 1 FP: 0 FN: 52 TN: 75
- epoch 8: auc=0.5706, f1=0.0000, recall=0.0000, score=0.1141  TP: 0 FP: 0 FN: 53 TN: 75
- epoch 7: auc=0.5605, f1=0.0000, recall=0.0000, score=0.1121  TP: 0 FP: 0 FN: 53 TN: 75
