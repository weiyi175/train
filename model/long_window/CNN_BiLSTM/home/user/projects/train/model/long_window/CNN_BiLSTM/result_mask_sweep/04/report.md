# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 50 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM//home/user/projects/train/model/long_window/CNN_BiLSTM/result_mask_sweep --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --mask_threshold 0.75 --mask_mode soft --window_mask_min_mean 0.65  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.5048
- F1: 0.0189
- Recall: 0.0095
## Confusion matrix (TP/FP/FN/TN)
- TP: 1
- FP: 0
- FN: 104
- TN: 151

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 1: auc=0.5000, f1=0.0000, recall=0.0000, score=0.1000  TP: 0 FP: 0 FN: 53 TN: 75
- epoch 2: auc=0.5000, f1=0.0000, recall=0.0000, score=0.1000  TP: 0 FP: 0 FN: 53 TN: 75
- epoch 3: auc=0.5000, f1=0.0000, recall=0.0000, score=0.1000  TP: 0 FP: 0 FN: 53 TN: 75
- epoch 4: auc=0.5000, f1=0.0000, recall=0.0000, score=0.1000  TP: 0 FP: 0 FN: 53 TN: 75
