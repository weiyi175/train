# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_sweep/cw02_fg03 --focal_alpha 0.2 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7112
- F1: 0.3453
- Recall: 0.2286
## Confusion matrix (TP/FP/FN/TN)
- TP: 24
- FP: 10
- FN: 81
- TN: 141

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 36: auc=0.7250, f1=0.6122, recall=0.5660, score=0.6117  TP: 30 FP: 15 FN: 23 TN: 60
- epoch 40: auc=0.7270, f1=0.6000, recall=0.5660, score=0.6084  TP: 30 FP: 17 FN: 23 TN: 58
- epoch 42: auc=0.7303, f1=0.5918, recall=0.5472, score=0.5972  TP: 29 FP: 16 FN: 24 TN: 59
- epoch 38: auc=0.7323, f1=0.5833, recall=0.5283, score=0.5856  TP: 28 FP: 15 FN: 25 TN: 60
