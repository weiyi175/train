# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw07_fg08 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.9 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7107
- F1: 0.5114
- Recall: 0.4286
## Confusion matrix (TP/FP/FN/TN)
- TP: 45
- FP: 26
- FN: 60
- TN: 125

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 51: auc=0.7723, f1=0.6408, recall=0.6226, score=0.6580  TP: 33 FP: 17 FN: 20 TN: 58
- epoch 49: auc=0.7507, f1=0.6154, recall=0.6038, score=0.6366  TP: 32 FP: 19 FN: 21 TN: 56
- epoch 33: auc=0.7439, f1=0.6154, recall=0.6038, score=0.6353  TP: 32 FP: 19 FN: 21 TN: 56
- epoch 42: auc=0.7623, f1=0.5859, recall=0.5472, score=0.6018  TP: 29 FP: 17 FN: 24 TN: 58
