# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw02_fg07 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7063
- F1: 0.5514
- Recall: 0.4857
## Confusion matrix (TP/FP/FN/TN)
- TP: 51
- FP: 29
- FN: 54
- TN: 122

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 36: auc=0.7036, f1=0.6325, recall=0.6981, score=0.6795  TP: 37 FP: 27 FN: 16 TN: 48
- epoch 40: auc=0.7155, f1=0.6261, recall=0.6792, score=0.6705  TP: 36 FP: 26 FN: 17 TN: 49
- epoch 38: auc=0.6966, f1=0.6316, recall=0.6792, score=0.6684  TP: 36 FP: 25 FN: 17 TN: 50
- epoch 29: auc=0.7147, f1=0.6422, recall=0.6604, score=0.6658  TP: 35 FP: 21 FN: 18 TN: 54
