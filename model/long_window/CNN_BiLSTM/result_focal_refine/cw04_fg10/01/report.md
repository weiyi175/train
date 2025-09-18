# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw04_fg10 --focal_alpha 0.35 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.95 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7465
- F1: 0.6010
- Recall: 0.5810
## Confusion matrix (TP/FP/FN/TN)
- TP: 61
- FP: 37
- FN: 44
- TN: 114

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 67: auc=0.7809, f1=0.6168, recall=0.6226, score=0.6525  TP: 33 FP: 21 FN: 20 TN: 54
- epoch 65: auc=0.7796, f1=0.6154, recall=0.6038, score=0.6424  TP: 32 FP: 19 FN: 21 TN: 56
- epoch 54: auc=0.7613, f1=0.5946, recall=0.6226, score=0.6420  TP: 33 FP: 25 FN: 20 TN: 50
- epoch 60: auc=0.7683, f1=0.6038, recall=0.6038, score=0.6367  TP: 32 FP: 21 FN: 21 TN: 54
