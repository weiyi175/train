# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw07_fg09 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.9 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7072
- F1: 0.5644
- Recall: 0.5429
## Confusion matrix (TP/FP/FN/TN)
- TP: 57
- FP: 40
- FN: 48
- TN: 111

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 48: auc=0.7097, f1=0.6241, recall=0.8302, score=0.7443  TP: 44 FP: 44 FN: 9 TN: 31
- epoch 34: auc=0.7374, f1=0.6441, recall=0.7170, score=0.6992  TP: 38 FP: 27 FN: 15 TN: 48
- epoch 20: auc=0.6843, f1=0.6154, recall=0.7547, score=0.6988  TP: 40 FP: 37 FN: 13 TN: 38
- epoch 52: auc=0.7180, f1=0.6190, recall=0.7358, score=0.6972  TP: 39 FP: 34 FN: 14 TN: 41
