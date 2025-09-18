# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw03_fg09 --focal_alpha 0.4 --focal_gamma_start 0.0 --focal_gamma_end 1.0 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.85 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7234
- F1: 0.5758
- Recall: 0.5429
## Confusion matrix (TP/FP/FN/TN)
- TP: 57
- FP: 36
- FN: 48
- TN: 115

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 25: auc=0.6948, f1=0.6000, recall=0.7925, score=0.7152  TP: 42 FP: 45 FN: 11 TN: 30
- epoch 41: auc=0.7444, f1=0.6240, recall=0.7358, score=0.7040  TP: 39 FP: 33 FN: 14 TN: 42
- epoch 31: auc=0.7233, f1=0.6387, recall=0.7170, score=0.6947  TP: 38 FP: 28 FN: 15 TN: 47
- epoch 27: auc=0.7358, f1=0.6491, recall=0.6981, score=0.6910  TP: 37 FP: 24 FN: 16 TN: 51
