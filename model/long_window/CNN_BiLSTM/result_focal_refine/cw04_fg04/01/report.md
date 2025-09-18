# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw04_fg04 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.95 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7219
- F1: 0.5567
- Recall: 0.5143
## Confusion matrix (TP/FP/FN/TN)
- TP: 54
- FP: 35
- FN: 51
- TN: 116

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 54: auc=0.7313, f1=0.6316, recall=0.6792, score=0.6754  TP: 36 FP: 25 FN: 17 TN: 50
- epoch 50: auc=0.7361, f1=0.6140, recall=0.6604, score=0.6616  TP: 35 FP: 26 FN: 18 TN: 49
- epoch 55: auc=0.7021, f1=0.5854, recall=0.6792, score=0.6557  TP: 36 FP: 34 FN: 17 TN: 41
- epoch 57: auc=0.7119, f1=0.5785, recall=0.6604, score=0.6461  TP: 35 FP: 33 FN: 18 TN: 42
