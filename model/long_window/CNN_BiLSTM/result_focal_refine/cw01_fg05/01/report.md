# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw01_fg05 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.75 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6930
- F1: 0.5217
- Recall: 0.4571
## Confusion matrix (TP/FP/FN/TN)
- TP: 48
- FP: 31
- FN: 57
- TN: 120

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 54: auc=0.7489, f1=0.6729, recall=0.6792, score=0.6913  TP: 36 FP: 18 FN: 17 TN: 57
- epoch 60: auc=0.7625, f1=0.6337, recall=0.6038, score=0.6445  TP: 32 FP: 16 FN: 21 TN: 59
- epoch 33: auc=0.7338, f1=0.6154, recall=0.6038, score=0.6333  TP: 32 FP: 19 FN: 21 TN: 56
- epoch 56: auc=0.7278, f1=0.5789, recall=0.6226, score=0.6306  TP: 33 FP: 28 FN: 20 TN: 47
