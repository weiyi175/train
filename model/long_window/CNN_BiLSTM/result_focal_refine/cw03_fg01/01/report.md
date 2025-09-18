# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw03_fg01 --focal_alpha 0.22 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.85 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6711
- F1: 0.0189
- Recall: 0.0095
## Confusion matrix (TP/FP/FN/TN)
- TP: 1
- FP: 0
- FN: 104
- TN: 151

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 15: auc=0.6038, f1=0.1724, recall=0.0943, score=0.2196  TP: 5 FP: 0 FN: 48 TN: 75
- epoch 16: auc=0.6033, f1=0.1724, recall=0.0943, score=0.2195  TP: 5 FP: 0 FN: 48 TN: 75
- epoch 14: auc=0.6005, f1=0.1071, recall=0.0566, score=0.1805  TP: 3 FP: 0 FN: 50 TN: 75
- epoch 13: auc=0.6008, f1=0.0727, recall=0.0377, score=0.1608  TP: 2 FP: 0 FN: 51 TN: 75
