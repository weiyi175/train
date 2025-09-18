# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_mask_sweep_60e/run_02 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --mask_threshold 0.75 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6977
- F1: 0.4196
- Recall: 0.2857
## Confusion matrix (TP/FP/FN/TN)
- TP: 30
- FP: 8
- FN: 75
- TN: 143

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 41: auc=0.7160, f1=0.5510, recall=0.5094, score=0.5632  TP: 27 FP: 18 FN: 26 TN: 57
- epoch 38: auc=0.7077, f1=0.5000, recall=0.4151, score=0.4991  TP: 22 FP: 13 FN: 31 TN: 62
- epoch 46: auc=0.7160, f1=0.4828, recall=0.3962, score=0.4861  TP: 21 FP: 13 FN: 32 TN: 62
- epoch 35: auc=0.6961, f1=0.4828, recall=0.3962, score=0.4822  TP: 21 FP: 13 FN: 32 TN: 62
