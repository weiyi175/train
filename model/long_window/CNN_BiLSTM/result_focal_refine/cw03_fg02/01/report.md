# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw03_fg02 --focal_alpha 0.22 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.85 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7048
- F1: 0.4663
- Recall: 0.3619
## Confusion matrix (TP/FP/FN/TN)
- TP: 38
- FP: 20
- FN: 67
- TN: 131

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 64: auc=0.7794, f1=0.6364, recall=0.6604, score=0.6770  TP: 35 FP: 22 FN: 18 TN: 53
- epoch 61: auc=0.7887, f1=0.6476, recall=0.6415, score=0.6728  TP: 34 FP: 18 FN: 19 TN: 57
- epoch 59: auc=0.7889, f1=0.6600, recall=0.6226, score=0.6671  TP: 33 FP: 14 FN: 20 TN: 61
- epoch 62: auc=0.7562, f1=0.5946, recall=0.6226, score=0.6409  TP: 33 FP: 25 FN: 20 TN: 50
