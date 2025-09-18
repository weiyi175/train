# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw04_fg05 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 0.95 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7479
- F1: 0.5650
- Recall: 0.4762
## Confusion matrix (TP/FP/FN/TN)
- TP: 50
- FP: 22
- FN: 55
- TN: 129

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 38: auc=0.6865, f1=0.5586, recall=0.5849, score=0.5973  TP: 31 FP: 27 FN: 22 TN: 48
- epoch 40: auc=0.7082, f1=0.5385, recall=0.5283, score=0.5673  TP: 28 FP: 23 FN: 25 TN: 52
- epoch 42: auc=0.7039, f1=0.5243, recall=0.5094, score=0.5528  TP: 27 FP: 23 FN: 26 TN: 52
- epoch 44: auc=0.7192, f1=0.5417, recall=0.4906, score=0.5516  TP: 26 FP: 17 FN: 27 TN: 58
