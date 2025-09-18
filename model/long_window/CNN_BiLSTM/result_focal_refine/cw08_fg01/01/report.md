# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw08_fg01 --focal_alpha 0.22 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7065
- F1: 0.2985
- Recall: 0.1905
## Confusion matrix (TP/FP/FN/TN)
- TP: 20
- FP: 9
- FN: 85
- TN: 142

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 30: auc=0.7255, f1=0.4524, recall=0.3585, score=0.4601  TP: 19 FP: 12 FN: 34 TN: 63
- epoch 33: auc=0.7275, f1=0.4286, recall=0.3396, score=0.4439  TP: 18 FP: 13 FN: 35 TN: 62
- epoch 36: auc=0.7260, f1=0.4198, recall=0.3208, score=0.4315  TP: 17 FP: 11 FN: 36 TN: 64
- epoch 38: auc=0.7230, f1=0.4000, recall=0.3208, score=0.4250  TP: 17 FP: 15 FN: 36 TN: 60
