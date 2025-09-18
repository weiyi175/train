# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw01_fg01 --focal_alpha 0.22 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.75 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7180
- F1: 0.4387
- Recall: 0.3238
## Confusion matrix (TP/FP/FN/TN)
- TP: 34
- FP: 16
- FN: 71
- TN: 135

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 43: auc=0.6933, f1=0.5556, recall=0.5660, score=0.5884  TP: 30 FP: 25 FN: 23 TN: 50
- epoch 41: auc=0.7097, f1=0.5657, recall=0.5283, score=0.5758  TP: 28 FP: 18 FN: 25 TN: 57
- epoch 36: auc=0.6999, f1=0.5106, recall=0.4528, score=0.5196  TP: 24 FP: 17 FN: 29 TN: 58
- epoch 39: auc=0.7117, f1=0.4835, recall=0.4151, score=0.4949  TP: 22 FP: 16 FN: 31 TN: 59
