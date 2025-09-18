# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw05_fg06 --focal_alpha 0.28 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.05 --class_weight_pos 0.8 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7227
- F1: 0.5189
- Recall: 0.4571
## Confusion matrix (TP/FP/FN/TN)
- TP: 48
- FP: 32
- FN: 57
- TN: 119

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 47: auc=0.7353, f1=0.6612, recall=0.7547, score=0.7228  TP: 40 FP: 28 FN: 13 TN: 47
- epoch 49: auc=0.7439, f1=0.6724, recall=0.7358, score=0.7184  TP: 39 FP: 24 FN: 14 TN: 51
- epoch 54: auc=0.7250, f1=0.6435, recall=0.6981, score=0.6871  TP: 37 FP: 25 FN: 16 TN: 50
- epoch 53: auc=0.7552, f1=0.6422, recall=0.6604, score=0.6739  TP: 35 FP: 21 FN: 18 TN: 54
