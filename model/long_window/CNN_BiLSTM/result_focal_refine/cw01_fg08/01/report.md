# CNN_BiLSTM Report

## Command
```
python train_cnn_bilstm.py --windows /home/user/projects/train/train_data/slipce/windows_npz.npz --epochs 70 --batch 16 --accumulate_steps 8 --result_dir /home/user/projects/train/model/long_window/CNN_BiLSTM/result_focal_refine/cw01_fg08 --focal_alpha 0.32 --focal_gamma_start 0.0 --focal_gamma_end 1.6 --curriculum_epochs 20 --run_seed None --class_weight_neg 1.0 --class_weight_pos 0.75 --mask_threshold 0.6 --mask_mode soft --window_mask_min_mean None  [auto-mask: occlusion_flag]
```

## Test metrics
- AUC: 0.7003
- F1: 0.5646
- Recall: 0.5619
## Confusion matrix (TP/FP/FN/TN)
- TP: 59
- FP: 45
- FN: 46
- TN: 106

## Top 4 epochs by score = 0.5 * recall + 0.3 * f1 + 0.2 * auc
- epoch 50: auc=0.7426, f1=0.6182, recall=0.6415, score=0.6547  TP: 34 FP: 23 FN: 19 TN: 52
- epoch 49: auc=0.7203, f1=0.5913, recall=0.6415, score=0.6422  TP: 34 FP: 28 FN: 19 TN: 47
- epoch 41: auc=0.6958, f1=0.5645, recall=0.6604, score=0.6387  TP: 35 FP: 36 FN: 18 TN: 39
- epoch 33: auc=0.6878, f1=0.5841, recall=0.6226, score=0.6241  TP: 33 FP: 27 FN: 20 TN: 48
