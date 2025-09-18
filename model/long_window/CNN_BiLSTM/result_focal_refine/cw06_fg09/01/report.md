# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw06_fg09 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.1 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7050
- F1: 0.5775
- Recall: 0.5143
## Confusion matrix (TP/FP/FN/TN)
- TP: 54
- FP: 28
- FN: 51
- TN: 123

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 49: auc=0.7203, f1=0.6269, recall=0.7925, score=0.7283  TP: 42 FP: 39 FN: 11 TN: 36
- epoch 47: auc=0.7331, f1=0.6357, recall=0.7736, score=0.7241  TP: 41 FP: 35 FN: 12 TN: 40
- epoch 42: auc=0.7210, f1=0.6260, recall=0.7736, score=0.7188  TP: 41 FP: 37 FN: 12 TN: 38
- epoch 45: auc=0.7301, f1=0.6202, recall=0.7547, score=0.7094  TP: 40 FP: 36 FN: 13 TN: 39
