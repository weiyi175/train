# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw08_fg03 --focal_alpha 0.25 --focal_gamma_start 0.0 --focal_gamma_end 1.4 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 1.0 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7404
- F1: 0.5089
- Recall: 0.4095
## Confusion matrix (TP/FP/FN/TN)
- TP: 43
- FP: 21
- FN: 62
- TN: 130

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 54: auc=0.7255, f1=0.5607, recall=0.5660, score=0.5964  TP: 30 FP: 24 FN: 23 TN: 51
- epoch 52: auc=0.7119, f1=0.5421, recall=0.5472, score=0.5786  TP: 29 FP: 25 FN: 24 TN: 50
- epoch 31: auc=0.7132, f1=0.5714, recall=0.5283, score=0.5782  TP: 28 FP: 17 FN: 25 TN: 58
- epoch 35: auc=0.7167, f1=0.5600, recall=0.5283, score=0.5755  TP: 28 FP: 19 FN: 25 TN: 56
