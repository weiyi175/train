# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw03_fg10 --focal_alpha 0.35 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.85 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7203
- F1: 0.4713
- Recall: 0.3524
## Confusion matrix (TP/FP/FN/TN)
- TP: 37
- FP: 15
- FN: 68
- TN: 136

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 28: auc=0.7313, f1=0.6667, recall=0.6226, score=0.6576  TP: 33 FP: 13 FN: 20 TN: 62
- epoch 27: auc=0.7323, f1=0.6531, recall=0.6038, score=0.6443  TP: 32 FP: 13 FN: 21 TN: 62
- epoch 31: auc=0.7228, f1=0.6337, recall=0.6038, score=0.6365  TP: 32 FP: 16 FN: 21 TN: 59
- epoch 25: auc=0.7313, f1=0.6327, recall=0.5849, score=0.6285  TP: 31 FP: 14 FN: 22 TN: 61
