# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw08_fg09 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7035
- F1: 0.5326
- Recall: 0.4667
## Confusion matrix (TP/FP/FN/TN)
- TP: 49
- FP: 30
- FN: 56
- TN: 121

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 25: auc=0.6579, f1=0.5536, recall=0.5849, score=0.5901  TP: 31 FP: 28 FN: 22 TN: 47
- epoch 21: auc=0.6714, f1=0.5660, recall=0.5660, score=0.5871  TP: 30 FP: 23 FN: 23 TN: 52
- epoch 27: auc=0.6971, f1=0.5743, recall=0.5472, score=0.5853  TP: 29 FP: 19 FN: 24 TN: 56
- epoch 26: auc=0.6875, f1=0.5545, recall=0.5283, score=0.5680  TP: 28 FP: 20 FN: 25 TN: 55
