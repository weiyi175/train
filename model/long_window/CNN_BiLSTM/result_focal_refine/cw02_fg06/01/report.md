# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw02_fg06 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6143
- F1: 0.0000
- Recall: 0.0000
## Confusion matrix (TP/FP/FN/TN)
- TP: 0
- FP: 0
- FN: 105
- TN: 151

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 10: auc=0.5836, f1=0.1053, recall=0.0566, score=0.1766  TP: 3 FP: 1 FN: 50 TN: 74
- epoch 9: auc=0.5799, f1=0.0364, recall=0.0189, score=0.1363  TP: 1 FP: 1 FN: 52 TN: 74
- epoch 1: auc=0.6101, f1=0.0000, recall=0.0000, score=0.1220  TP: 0 FP: 0 FN: 53 TN: 75
- epoch 11: auc=0.6101, f1=0.0000, recall=0.0000, score=0.1220  TP: 0 FP: 0 FN: 53 TN: 75
