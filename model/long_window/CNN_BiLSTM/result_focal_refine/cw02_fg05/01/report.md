# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw02_fg05 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7371
- F1: 0.5941
- Recall: 0.5714
## Confusion matrix (TP/FP/FN/TN)
- TP: 60
- FP: 37
- FN: 45
- TN: 114

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 35: auc=0.7514, f1=0.6182, recall=0.6415, score=0.6565  TP: 34 FP: 23 FN: 19 TN: 52
- epoch 45: auc=0.7514, f1=0.6182, recall=0.6415, score=0.6565  TP: 34 FP: 23 FN: 19 TN: 52
- epoch 37: auc=0.7404, f1=0.6071, recall=0.6415, score=0.6510  TP: 34 FP: 25 FN: 19 TN: 50
- epoch 40: auc=0.7165, f1=0.5913, recall=0.6415, score=0.6414  TP: 34 FP: 28 FN: 19 TN: 47
