# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw06_fg10 --focal_alpha 0.35 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.1 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6689
- F1: 0.4348
- Recall: 0.3333
## Confusion matrix (TP/FP/FN/TN)
- TP: 35
- FP: 21
- FN: 70
- TN: 130

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 40: auc=0.7517, f1=0.6364, recall=0.6604, score=0.6714  TP: 35 FP: 22 FN: 18 TN: 53
- epoch 38: auc=0.7592, f1=0.6355, recall=0.6415, score=0.6633  TP: 34 FP: 20 FN: 19 TN: 55
- epoch 42: auc=0.7517, f1=0.6355, recall=0.6415, score=0.6617  TP: 34 FP: 20 FN: 19 TN: 55
- epoch 35: auc=0.7482, f1=0.6355, recall=0.6415, score=0.6610  TP: 34 FP: 20 FN: 19 TN: 55
