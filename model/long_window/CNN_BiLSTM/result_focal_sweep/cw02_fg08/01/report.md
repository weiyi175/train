# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw02_fg08 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7472
- F1: 0.5960
- Recall: 0.5619
## Confusion matrix (TP/FP/FN/TN)
- TP: 59
- FP: 34
- FN: 46
- TN: 117

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 44: auc=0.7457, f1=0.6126, recall=0.6415, score=0.6537  TP: 34 FP: 24 FN: 19 TN: 51
- epoch 33: auc=0.7245, f1=0.5913, recall=0.6415, score=0.6431  TP: 34 FP: 28 FN: 19 TN: 47
- epoch 42: auc=0.7182, f1=0.5812, recall=0.6415, score=0.6388  TP: 34 FP: 30 FN: 19 TN: 45
- epoch 58: auc=0.7436, f1=0.5926, recall=0.6038, score=0.6284  TP: 32 FP: 23 FN: 21 TN: 52
