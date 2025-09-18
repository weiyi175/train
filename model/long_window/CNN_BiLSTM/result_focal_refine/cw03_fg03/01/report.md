# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw03_fg03 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.85 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6525
- F1: 0.0189
- Recall: 0.0095
## Confusion matrix (TP/FP/FN/TN)
- TP: 1
- FP: 0
- FN: 104
- TN: 151

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 9: auc=0.5839, f1=0.1404, recall=0.0755, score=0.1966  TP: 4 FP: 0 FN: 49 TN: 75
- epoch 10: auc=0.5831, f1=0.1404, recall=0.0755, score=0.1965  TP: 4 FP: 0 FN: 49 TN: 75
- epoch 8: auc=0.5894, f1=0.0727, recall=0.0377, score=0.1586  TP: 2 FP: 0 FN: 51 TN: 75
- epoch 1: auc=0.6639, f1=0.0000, recall=0.0000, score=0.1328  TP: 0 FP: 0 FN: 53 TN: 75
