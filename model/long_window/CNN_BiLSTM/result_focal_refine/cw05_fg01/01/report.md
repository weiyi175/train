# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw05_fg01 --focal_alpha 0.22 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7540
- F1: 0.5938
- Recall: 0.5429
## Confusion matrix (TP/FP/FN/TN)
- TP: 57
- FP: 30
- FN: 48
- TN: 121

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 35: auc=0.6938, f1=0.5532, recall=0.4906, score=0.5500  TP: 26 FP: 15 FN: 27 TN: 60
- epoch 64: auc=0.7223, f1=0.5376, recall=0.4717, score=0.5416  TP: 25 FP: 15 FN: 28 TN: 60
- epoch 53: auc=0.7278, f1=0.5319, recall=0.4717, score=0.5410  TP: 25 FP: 16 FN: 28 TN: 59
- epoch 66: auc=0.7281, f1=0.5053, recall=0.4528, score=0.5236  TP: 24 FP: 18 FN: 29 TN: 57
