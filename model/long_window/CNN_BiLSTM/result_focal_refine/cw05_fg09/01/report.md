# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw05_fg09 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7347
- F1: 0.6406
- Recall: 0.7810
## Confusion matrix (TP/FP/FN/TN)
- TP: 82
- FP: 69
- FN: 23
- TN: 82

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 70: auc=0.7688, f1=0.6777, recall=0.7736, score=0.7439  TP: 41 FP: 27 FN: 12 TN: 48
- epoch 61: auc=0.7741, f1=0.6780, recall=0.7547, score=0.7356  TP: 40 FP: 25 FN: 13 TN: 50
- epoch 66: auc=0.7570, f1=0.6838, recall=0.7547, score=0.7339  TP: 40 FP: 24 FN: 13 TN: 51
- epoch 46: auc=0.7494, f1=0.6612, recall=0.7547, score=0.7256  TP: 40 FP: 28 FN: 13 TN: 47
