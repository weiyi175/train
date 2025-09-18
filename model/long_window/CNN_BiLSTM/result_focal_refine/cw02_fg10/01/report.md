# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw02_fg10 --focal_alpha 0.35 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7008
- F1: 0.4744
- Recall: 0.3524
## Confusion matrix (TP/FP/FN/TN)
- TP: 37
- FP: 14
- FN: 68
- TN: 137

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 66: auc=0.7562, f1=0.6055, recall=0.6226, score=0.6442  TP: 33 FP: 23 FN: 20 TN: 52
- epoch 62: auc=0.7623, f1=0.6038, recall=0.6038, score=0.6355  TP: 32 FP: 21 FN: 21 TN: 54
- epoch 69: auc=0.7992, f1=0.6061, recall=0.5660, score=0.6247  TP: 30 FP: 16 FN: 23 TN: 59
- epoch 29: auc=0.7316, f1=0.6019, recall=0.5849, score=0.6193  TP: 31 FP: 19 FN: 22 TN: 56
