# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 100 --batch 64 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.5 --curriculum_epochs 20 --run_seed None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7494
- F1: 0.6010
- Recall: 0.5524
## Confusion matrix (TP/FP/FN/TN)
- TP: 58
- FP: 30
- FN: 47
- TN: 121

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 38: auc=0.7484, f1=0.6481, recall=0.6604, score=0.6743  TP: 35 FP: 20 FN: 18 TN: 55
- epoch 42: auc=0.7499, f1=0.6355, recall=0.6415, score=0.6614  TP: 34 FP: 20 FN: 19 TN: 55
- epoch 40: auc=0.7492, f1=0.6535, recall=0.6226, score=0.6572  TP: 33 FP: 15 FN: 20 TN: 60
- epoch 36: auc=0.7462, f1=0.6346, recall=0.6226, score=0.6509  TP: 33 FP: 18 FN: 20 TN: 57
