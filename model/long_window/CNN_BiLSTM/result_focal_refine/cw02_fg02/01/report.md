# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw02_fg02 --focal_alpha 0.22 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7217
- F1: 0.3597
- Recall: 0.2381
## Confusion matrix (TP/FP/FN/TN)
- TP: 25
- FP: 9
- FN: 80
- TN: 142

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 37: auc=0.7014, f1=0.5657, recall=0.5283, score=0.5741  TP: 28 FP: 18 FN: 25 TN: 57
- epoch 33: auc=0.6991, f1=0.5106, recall=0.4528, score=0.5194  TP: 24 FP: 17 FN: 29 TN: 58
- epoch 35: auc=0.6948, f1=0.5055, recall=0.4340, score=0.5076  TP: 23 FP: 15 FN: 30 TN: 60
- epoch 31: auc=0.6780, f1=0.4946, recall=0.4340, score=0.5010  TP: 23 FP: 17 FN: 30 TN: 58
