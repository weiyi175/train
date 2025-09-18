# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw05_fg05 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.6985
- F1: 0.4762
- Recall: 0.3810
## Confusion matrix (TP/FP/FN/TN)
- TP: 40
- FP: 23
- FN: 65
- TN: 128

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 39: auc=0.7308, f1=0.6018, recall=0.6415, score=0.6474  TP: 34 FP: 26 FN: 19 TN: 49
- epoch 41: auc=0.7238, f1=0.5714, recall=0.6038, score=0.6181  TP: 32 FP: 27 FN: 21 TN: 48
- epoch 37: auc=0.7147, f1=0.5556, recall=0.5660, score=0.5926  TP: 30 FP: 25 FN: 23 TN: 50
- epoch 32: auc=0.7233, f1=0.5417, recall=0.4906, score=0.5524  TP: 26 FP: 17 FN: 27 TN: 58
