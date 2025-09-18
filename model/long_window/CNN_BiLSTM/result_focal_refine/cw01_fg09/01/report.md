# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw01_fg09 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.75 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7335
- F1: 0.5150
- Recall: 0.4095
## Confusion matrix (TP/FP/FN/TN)
- TP: 43
- FP: 19
- FN: 62
- TN: 132

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 45: auc=0.7436, f1=0.5905, recall=0.5849, score=0.6183  TP: 31 FP: 21 FN: 22 TN: 54
- epoch 44: auc=0.7469, f1=0.6122, recall=0.5660, score=0.6161  TP: 30 FP: 15 FN: 23 TN: 60
- epoch 28: auc=0.7187, f1=0.5905, recall=0.5849, score=0.6133  TP: 31 FP: 21 FN: 22 TN: 54
- epoch 36: auc=0.7328, f1=0.5941, recall=0.5660, score=0.6078  TP: 30 FP: 18 FN: 23 TN: 57
