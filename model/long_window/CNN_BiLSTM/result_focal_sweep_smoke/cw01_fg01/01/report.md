# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 10 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep_smoke/cw01_fg01 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6532
- F1: 0.1379
- Recall: 0.0762
## Confusion matrix (TP/FP/FN/TN)
- TP: 8
- FP: 3
- FN: 97
- TN: 148

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 10: auc=0.6131, f1=0.1071, recall=0.0566, score=0.1831  TP: 3 FP: 0 FN: 50 TN: 75
- epoch 9: auc=0.6101, f1=0.0727, recall=0.0377, score=0.1627  TP: 2 FP: 0 FN: 51 TN: 75
- epoch 8: auc=0.6078, f1=0.0714, recall=0.0377, score=0.1619  TP: 2 FP: 1 FN: 51 TN: 74
- epoch 6: auc=0.6055, f1=0.0702, recall=0.0377, score=0.1610  TP: 2 FP: 2 FN: 51 TN: 73
