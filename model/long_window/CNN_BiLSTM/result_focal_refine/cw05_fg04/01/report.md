# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw05_fg04 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7017
- F1: 0.2985
- Recall: 0.1905
## Confusion matrix (TP/FP/FN/TN)
- TP: 20
- FP: 9
- FN: 85
- TN: 142

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 37: auc=0.7353, f1=0.4773, recall=0.3962, score=0.4884  TP: 21 FP: 14 FN: 32 TN: 61
- epoch 40: auc=0.7381, f1=0.4598, recall=0.3774, score=0.4742  TP: 20 FP: 14 FN: 33 TN: 61
- epoch 39: auc=0.7457, f1=0.4419, recall=0.3585, score=0.4609  TP: 19 FP: 14 FN: 34 TN: 61
- epoch 41: auc=0.7396, f1=0.4444, recall=0.3396, score=0.4511  TP: 18 FP: 10 FN: 35 TN: 65
