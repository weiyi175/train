# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw06_fg08 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.1 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6994
- F1: 0.3611
- Recall: 0.2476
## Confusion matrix (TP/FP/FN/TN)
- TP: 26
- FP: 13
- FN: 79
- TN: 138

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 60: auc=0.7889, f1=0.6789, recall=0.6981, score=0.7105  TP: 37 FP: 19 FN: 16 TN: 56
- epoch 69: auc=0.7658, f1=0.6491, recall=0.6981, score=0.6970  TP: 37 FP: 24 FN: 16 TN: 51
- epoch 66: auc=0.7487, f1=0.6080, recall=0.7170, score=0.6906  TP: 38 FP: 34 FN: 15 TN: 41
- epoch 41: auc=0.7316, f1=0.6316, recall=0.6792, score=0.6754  TP: 36 FP: 25 FN: 17 TN: 50
