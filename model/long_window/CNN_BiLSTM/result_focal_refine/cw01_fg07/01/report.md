# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw01_fg07 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.75 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7470
- F1: 0.4557
- Recall: 0.3429
## Confusion matrix (TP/FP/FN/TN)
- TP: 36
- FP: 17
- FN: 69
- TN: 134

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 27: auc=0.7112, f1=0.5833, recall=0.5283, score=0.5814  TP: 28 FP: 15 FN: 25 TN: 60
- epoch 25: auc=0.7223, f1=0.5556, recall=0.4717, score=0.5470  TP: 25 FP: 12 FN: 28 TN: 63
- epoch 33: auc=0.6913, f1=0.5435, recall=0.4717, score=0.5372  TP: 25 FP: 14 FN: 28 TN: 61
- epoch 32: auc=0.7132, f1=0.5217, recall=0.4528, score=0.5256  TP: 24 FP: 15 FN: 29 TN: 60
