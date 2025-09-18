# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 100 --batch 64 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --no_early_stop --run_seed 30
```

## Test metrics
- AUC: 0.7463
- F1: 0.5325
- Recall: 0.4286
## Confusion matrix (TP/FP/FN/TN)
- TP: 45
- FP: 19
- FN: 60
- TN: 132

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 93: auc=0.7238, f1=0.6422, recall=0.6604, score=0.6676  TP: 35 FP: 21 FN: 18 TN: 54
- epoch 92: auc=0.7291, f1=0.6346, recall=0.6226, score=0.6475  TP: 33 FP: 18 FN: 20 TN: 57
- epoch 38: auc=0.6755, f1=0.5833, recall=0.6604, score=0.6403  TP: 35 FP: 32 FN: 18 TN: 43
- epoch 86: auc=0.7419, f1=0.5946, recall=0.6226, score=0.6381  TP: 33 FP: 25 FN: 20 TN: 50
