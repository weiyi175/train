# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw01_fg03 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.75 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7082
- F1: 0.4472
- Recall: 0.3429
## Confusion matrix (TP/FP/FN/TN)
- TP: 36
- FP: 20
- FN: 69
- TN: 131

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 42: auc=0.7338, f1=0.5376, recall=0.4717, score=0.5439  TP: 25 FP: 15 FN: 28 TN: 60
- epoch 39: auc=0.7313, f1=0.4944, recall=0.4151, score=0.5021  TP: 22 FP: 14 FN: 31 TN: 61
- epoch 35: auc=0.7459, f1=0.4828, recall=0.3962, score=0.4921  TP: 21 FP: 13 FN: 32 TN: 62
- epoch 38: auc=0.7484, f1=0.4773, recall=0.3962, score=0.4910  TP: 21 FP: 14 FN: 32 TN: 61
