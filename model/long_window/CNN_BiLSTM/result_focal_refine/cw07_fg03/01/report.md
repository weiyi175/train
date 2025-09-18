# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw07_fg03 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.9 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7470
- F1: 0.5851
- Recall: 0.5238
## Confusion matrix (TP/FP/FN/TN)
- TP: 55
- FP: 28
- FN: 50
- TN: 123

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 51: auc=0.7517, f1=0.6667, recall=0.6981, score=0.6994  TP: 37 FP: 21 FN: 16 TN: 54
- epoch 53: auc=0.7343, f1=0.6542, recall=0.6604, score=0.6733  TP: 35 FP: 19 FN: 18 TN: 56
- epoch 49: auc=0.7233, f1=0.6154, recall=0.6792, score=0.6689  TP: 36 FP: 28 FN: 17 TN: 47
- epoch 47: auc=0.7396, f1=0.6602, recall=0.6415, score=0.6667  TP: 34 FP: 16 FN: 19 TN: 59
