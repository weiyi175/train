# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw05_fg01 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.8 --class_weight_pos 1.2 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7306
- F1: 0.5574
- Recall: 0.4857
## Confusion matrix (TP/FP/FN/TN)
- TP: 51
- FP: 27
- FN: 54
- TN: 124

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 58: auc=0.7706, f1=0.6435, recall=0.6981, score=0.6962  TP: 37 FP: 25 FN: 16 TN: 50
- epoch 56: auc=0.7708, f1=0.6429, recall=0.6792, score=0.6866  TP: 36 FP: 23 FN: 17 TN: 52
- epoch 54: auc=0.7824, f1=0.6214, recall=0.6038, score=0.6448  TP: 32 FP: 18 FN: 21 TN: 57
- epoch 64: auc=0.7824, f1=0.6214, recall=0.6038, score=0.6448  TP: 32 FP: 18 FN: 21 TN: 57
