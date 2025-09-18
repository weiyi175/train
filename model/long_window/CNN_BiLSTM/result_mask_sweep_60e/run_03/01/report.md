# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 60 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_mask_sweep_60e/run_03 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None --mask_threshold 0.8 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6539
- F1: 0.3053
- Recall: 0.1905
## Confusion matrix (TP/FP/FN/TN)
- TP: 20
- FP: 6
- FN: 85
- TN: 145

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 36: auc=0.7072, f1=0.4762, recall=0.3774, score=0.4730  TP: 20 FP: 11 FN: 33 TN: 64
- epoch 38: auc=0.7117, f1=0.4578, recall=0.3585, score=0.4589  TP: 19 FP: 11 FN: 34 TN: 64
- epoch 39: auc=0.6999, f1=0.4524, recall=0.3585, score=0.4549  TP: 19 FP: 12 FN: 34 TN: 63
- epoch 37: auc=0.7140, f1=0.4250, recall=0.3208, score=0.4307  TP: 17 FP: 10 FN: 36 TN: 65
