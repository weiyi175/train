# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw02_fg04 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7036
- F1: 0.3576
- Recall: 0.2571
## Confusion matrix (TP/FP/FN/TN)
- TP: 27
- FP: 19
- FN: 78
- TN: 132

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 40: auc=0.7165, f1=0.6126, recall=0.6415, score=0.6478  TP: 34 FP: 24 FN: 19 TN: 51
- epoch 37: auc=0.6996, f1=0.5962, recall=0.5849, score=0.6112  TP: 31 FP: 20 FN: 22 TN: 55
- epoch 39: auc=0.6976, f1=0.5741, recall=0.5849, score=0.6042  TP: 31 FP: 24 FN: 22 TN: 51
- epoch 38: auc=0.7059, f1=0.5882, recall=0.5660, score=0.6007  TP: 30 FP: 19 FN: 23 TN: 56
