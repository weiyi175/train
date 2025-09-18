# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw07_fg10 --focal_alpha 0.35 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.9 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7155
- F1: 0.5913
- Recall: 0.6476
## Confusion matrix (TP/FP/FN/TN)
- TP: 68
- FP: 57
- FN: 37
- TN: 94

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 65: auc=0.6823, f1=0.6111, recall=0.8302, score=0.7349  TP: 44 FP: 47 FN: 9 TN: 28
- epoch 67: auc=0.7180, f1=0.6187, recall=0.8113, score=0.7349  TP: 43 FP: 43 FN: 10 TN: 32
- epoch 69: auc=0.7464, f1=0.6560, recall=0.7736, score=0.7329  TP: 41 FP: 31 FN: 12 TN: 44
- epoch 62: auc=0.7107, f1=0.6056, recall=0.8113, score=0.7295  TP: 43 FP: 46 FN: 10 TN: 29
