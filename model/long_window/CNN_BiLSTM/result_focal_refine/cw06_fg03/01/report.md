# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw06_fg03 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.1 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7304
- F1: 0.5281
- Recall: 0.4476
## Confusion matrix (TP/FP/FN/TN)
- TP: 47
- FP: 26
- FN: 58
- TN: 125

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 37: auc=0.6883, f1=0.5577, recall=0.5472, score=0.5786  TP: 29 FP: 22 FN: 24 TN: 53
- epoch 39: auc=0.7117, f1=0.5243, recall=0.5094, score=0.5543  TP: 27 FP: 23 FN: 26 TN: 52
- epoch 53: auc=0.6918, f1=0.5143, recall=0.5094, score=0.5474  TP: 27 FP: 25 FN: 26 TN: 50
- epoch 48: auc=0.6747, f1=0.4860, recall=0.4906, score=0.5260  TP: 26 FP: 28 FN: 27 TN: 47
