# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw05_fg08 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6968
- F1: 0.4947
- Recall: 0.4476
## Confusion matrix (TP/FP/FN/TN)
- TP: 47
- FP: 38
- FN: 58
- TN: 113

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 34: auc=0.6961, f1=0.6107, recall=0.7547, score=0.6998  TP: 40 FP: 38 FN: 13 TN: 37
- epoch 36: auc=0.7062, f1=0.6032, recall=0.7170, score=0.6807  TP: 38 FP: 35 FN: 15 TN: 40
- epoch 32: auc=0.6964, f1=0.5983, recall=0.6604, score=0.6489  TP: 35 FP: 29 FN: 18 TN: 46
- epoch 30: auc=0.7094, f1=0.5965, recall=0.6415, score=0.6416  TP: 34 FP: 27 FN: 19 TN: 48
