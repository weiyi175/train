# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir result_confirm --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --no_early_stop --run_seed None --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7005
- F1: 0.5584
- Recall: 0.5238
## Confusion matrix (TP/FP/FN/TN)
- TP: 55
- FP: 37
- FN: 50
- TN: 114

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 65: auc=0.7371, f1=0.6491, recall=0.6981, score=0.6912  TP: 37 FP: 24 FN: 16 TN: 51
- epoch 63: auc=0.7240, f1=0.6435, recall=0.6981, score=0.6869  TP: 37 FP: 25 FN: 16 TN: 50
- epoch 68: auc=0.7489, f1=0.6429, recall=0.6792, score=0.6823  TP: 36 FP: 23 FN: 17 TN: 52
- epoch 70: auc=0.7613, f1=0.6476, recall=0.6415, score=0.6673  TP: 34 FP: 18 FN: 19 TN: 57
