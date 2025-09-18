# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw02_fg01 --focal_alpha 0.22 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6991
- F1: 0.4804
- Recall: 0.4095
## Confusion matrix (TP/FP/FN/TN)
- TP: 43
- FP: 31
- FN: 62
- TN: 120

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 59: auc=0.7706, f1=0.6000, recall=0.5660, score=0.6171  TP: 30 FP: 17 FN: 23 TN: 58
- epoch 51: auc=0.7640, f1=0.6000, recall=0.5660, score=0.6158  TP: 30 FP: 17 FN: 23 TN: 58
- epoch 57: auc=0.7872, f1=0.6105, recall=0.5472, score=0.6142  TP: 29 FP: 13 FN: 24 TN: 62
- epoch 46: auc=0.7562, f1=0.5773, recall=0.5283, score=0.5886  TP: 28 FP: 16 FN: 25 TN: 59
