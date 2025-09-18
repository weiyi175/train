# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw03_fg04 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.85 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6960
- F1: 0.5031
- Recall: 0.3905
## Confusion matrix (TP/FP/FN/TN)
- TP: 41
- FP: 17
- FN: 64
- TN: 134

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 40: auc=0.6953, f1=0.6116, recall=0.6981, score=0.6716  TP: 37 FP: 31 FN: 16 TN: 44
- epoch 45: auc=0.7109, f1=0.5812, recall=0.6415, score=0.6373  TP: 34 FP: 30 FN: 19 TN: 45
- epoch 38: auc=0.6989, f1=0.5714, recall=0.6038, score=0.6131  TP: 32 FP: 27 FN: 21 TN: 48
- epoch 50: auc=0.6918, f1=0.5517, recall=0.6038, score=0.6058  TP: 32 FP: 31 FN: 21 TN: 44
