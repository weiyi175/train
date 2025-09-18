# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw05_fg10 --focal_alpha 0.35 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7425
- F1: 0.5227
- Recall: 0.4381
## Confusion matrix (TP/FP/FN/TN)
- TP: 46
- FP: 25
- FN: 59
- TN: 126

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 46: auc=0.7811, f1=0.6606, recall=0.6792, score=0.6940  TP: 36 FP: 20 FN: 17 TN: 55
- epoch 51: auc=0.7701, f1=0.6372, recall=0.6792, score=0.6848  TP: 36 FP: 24 FN: 17 TN: 51
- epoch 49: auc=0.7678, f1=0.6364, recall=0.6604, score=0.6747  TP: 35 FP: 22 FN: 18 TN: 53
- epoch 36: auc=0.7597, f1=0.6337, recall=0.6038, score=0.6439  TP: 32 FP: 16 FN: 21 TN: 59
